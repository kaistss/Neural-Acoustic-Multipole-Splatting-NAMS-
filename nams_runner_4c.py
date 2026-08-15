import os
import argparse
from shutil import copyfile
import yaml
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets_loader import WaveLoader
from model_4c import NAMS_network
from renderer_4c import  NAMS_renderer
from utils.metric import metric_cal
from utils.logger import logger_config, logging_loss_4c
from utils.criterion_4c import Criterion

SEED = 2025

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)


class NAMS_Runner():
    def __init__(self, **kwargs) -> None:

        kwargs_path = kwargs['path']
        kwargs_render = kwargs['render']
        kwargs_network = kwargs['model']
        kwargs_train = kwargs['train']

        ## Path settings
        self.dataset_type = kwargs_path['dataset_type']
        self.dataset_dir = kwargs_path['dataset_dir']
        self.logdir = kwargs_path['logdir']

        ## Logger
        log_filename = "logger.log"
        log_savepath = os.path.join(self.logdir, log_filename)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='nams')
        self.logger.info("data type:%s, logdir:%s", self.dataset_type, self.logdir)

        ## tensorboard writer
        log_prefix = f'tensorboard_logs/{(self.logdir).split("/")[1]}'
        os.makedirs(log_prefix, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f'{log_prefix}')

        # Train settings
        # epochs settings
        self.kwargs_train = kwargs['train']
        self.batch_size = kwargs_train['batch_size']
        self.epochs = kwargs_train['epochs']
        self.val_epochs = kwargs_train['val_epochs']
        self.save_epochs = kwargs_train['save_epochs']
        self.prune_epochs = kwargs_train['prune_epochs']
        self.prune_start_epoch = kwargs_train['prune_start_epoch']
        self.current_epoch = 0
        self.current_iteration = 0 

        # optimizer. scheduler settings
        self.lr = float(kwargs_train['lr'])
        self.lr_prior = float(kwargs_train['lr_prior'])
        self.weight_decay = kwargs_train['weight_decay']
        self.T_max = kwargs_train['T_max']
        self.eta_min = float(kwargs_train['eta_min'])

        ## dataloader
        self.ir_length = kwargs_render['ir_length']
        self.fs = kwargs_render['fs']
        self.n_collocation = kwargs_train.get('n_collocation', 32)
        self.train_set = WaveLoader(base_folder=self.dataset_dir, dataset_type=self.dataset_type, eval=False, seq_len=int(self.ir_length*self.fs), fs=self.fs, n_collocation=self.n_collocation)
        self.test_set  = WaveLoader(base_folder=self.dataset_dir, dataset_type=self.dataset_type, eval=True, seq_len=int(self.ir_length*self.fs), fs=self.fs, n_collocation=self.n_collocation)
    
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.logger.info("Train set size:%d, Test set size:%d", len(self.train_set), len(self.test_set))

        ## speaker position (fixed)
        self.spk_pos = self.train_set.positions_tx[0]

        ## network, renderer
        self.network = NAMS_network(kwargs_network, kwargs_render) # network
        self.renderer = NAMS_renderer(networks_fn=self.network, spk_pos=self.spk_pos, **kwargs_render) # renderer

        # multi gpu
        if torch.cuda.device_count() > 1: self.renderer = nn.DataParallel(self.renderer)
        self.renderer = self.renderer.cuda()
        # the unwrapped module: DataParallel only forwards forward(), so evaluate_field()/pruning()
        # and the state_dict have to go through the underlying renderer
        self.renderer_module = self.renderer.module if isinstance(self.renderer, nn.DataParallel) else self.renderer

        ## Learnable Weights
        ## log-parameterized so that sigma = exp(log_sigma) stays strictly positive:
        ## optimizing sigma directly lets it cross zero (log(sigma) -> NaN, 1/sigma^2 -> inf)
        self.log_sigma = []
        self.log_sigma.append(torch.tensor(np.log(2/kwargs_train['init_eps_data']), dtype=torch.float32, device="cuda", requires_grad=True))
        self.log_sigma.append(torch.tensor(np.log(2/kwargs_train['init_eps_prior']), dtype=torch.float32, device="cuda", requires_grad=True))
        ## Optimizser, Scheduler
        self.optimizer = torch.optim.Adam(self.renderer.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.prior_optimizer = torch.optim.Adam(self.log_sigma, lr=self.lr_prior, betas=(0.9, 0.9), weight_decay=1e-5)
        self.prior_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.prior_optimizer, T_max=self.T_max, eta_min = self.eta_min)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.T_max, eta_min=self.eta_min, last_epoch=-1)



        
        ## Print total number of parameters
        params = list(self.renderer.parameters())
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.logger.info("Total number of parameters: %s", total_params)

        # loss settings
        self.criterion = Criterion(kwargs_train)
        self.best_valid_loss = 1e10

    def uncertainty_weighted_loss(self, spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, momentum_loss, continuity_loss):
        """ PI-DANF style homoscedastic uncertainty weighting.
        loss = sum_i L_i / (2 sigma_i^2) + sum_i log(sigma_i), with sigma_i = exp(log_sigma_i)
        """
        data_loss = (spec_loss + amplitude_loss + angle_loss + time_loss + energy_loss + multi_stft_loss) * torch.exp(-2 * self.log_sigma[0]) / 2
        physics_loss = (momentum_loss + continuity_loss + 1) * torch.exp(-2 * self.log_sigma[1]) / 2
        total_loss = data_loss + physics_loss + self.log_sigma[0] + self.log_sigma[1]
        return total_loss, data_loss, physics_loss

    @staticmethod
    def sanitize_grads(tensors):
        with torch.no_grad():
            for tensor in tensors:
                if tensor.grad is not None:
                    torch.nan_to_num_(tensor.grad, nan=0.0, posinf=0.0, neginf=0.0)

    def save_checkpoint(self, mode='period'):
        ckptsdir = os.path.join(self.logdir, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)

        if mode == 'best':
            ckptname = os.path.join(ckptsdir, 'best.tar')
        else:
            ckptname = os.path.join(ckptsdir, '{:04d}.tar'.format(self.current_epoch))

        multipoles_count = self.renderer_module.get_count_mps()
        state_dict = self.renderer_module.state_dict()

        torch.save({
            'current_epoch': self.current_epoch,
            'multipoles_count': multipoles_count,
            'renderer_state_dict': state_dict,
            'log_sigma': [s.detach().cpu() for s in self.log_sigma],
            'optimizer_state_dict': self.optimizer.state_dict(),
            'prior_optimizer_state_dict': self.prior_optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, ckptname)

        self.logger.info(f"multipoles_count: {multipoles_count}")
        return ckptname



    def train(self):
        self.logger.info(f"Start training. Current epoch:{self.current_epoch}")

        while self.current_epoch < self.epochs:
            self.current_epoch += 1

            ## Logging multipoles count
            self.logger.info(f"Epochs: {self.current_epoch}, multipoles count:{self.renderer_module.get_count_mps()}")

            pbar = tqdm(total=len(self.train_loader), desc=f" Epoch: {self.current_epoch}/{self.epochs}", leave=True, position=0)
            self.current_iteration = 0

            train_losses = {'total_loss': 0, 'spec_loss': 0, 'amplitude_loss': 0, 'angle_loss': 0, 'time_loss': 0, 'energy_loss': 0, 'multi_stft_loss': 0, 'momentum_loss': 0, 'continuity_loss': 0, 'data_loss': 0, 'physical_loss': 0}
            for train_batch in self.train_loader:
                self.current_iteration += 1

                ## GT 
                ori_sig, position_rx, position_tx = train_batch
                coords_pde = self.train_set._sample_collocation_points(self.train_set.n_collocation)
                coords_pde = torch.from_numpy(coords_pde).cuda()

                ## Pred
                pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda()) # [B, F, 4, 2]
                pred_sig = pred_sig[...,0] + 1j * pred_sig[...,1]                # [B, F, 4]
                ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)                    # [B, F, 4]

                ## Loss calculation
                spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, momentum_loss, continuity_loss, _, _ = self.criterion(self.renderer_module, coords_pde, position_tx.cuda(), pred_sig, ori_sig)
                ## Total loss - 이것도 PI-DANF의 trainable loss 따라하기
                total_loss, data_loss, physics_loss = self.uncertainty_weighted_loss(
                    spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, momentum_loss, continuity_loss)
                # 임시로 데이터 로스만 보도록 수정하기
                # total_loss = spec_loss + amplitude_loss + angle_loss + time_loss + energy_loss + multi_stft_loss
                ## Backward
                self.optimizer.zero_grad()
                self.prior_optimizer.zero_grad()
                total_loss.backward()

                ## Sanitize NaN/Inf first: clipping before this turns a single NaN grad into
                ## a NaN total-norm, which poisons every other gradient
                self.sanitize_grads(list(self.renderer.parameters()) + self.log_sigma)
                ## Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.renderer.parameters(), max_norm=1)
                ## Optimizer step
                self.optimizer.step()
                self.prior_optimizer.step()
                
                ## Logging
                losses = logging_loss_4c(total_loss, spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, momentum_loss, continuity_loss, data_loss, physics_loss)
                for key in train_losses:
                    train_losses[key] += losses[key].detach()
                pbar.update(1)
                pbar.set_postfix_str('loss = {:.4f}, multi stft loss:{:.4f}, spec loss:{:.3f}, time loss:{:.3f}, energy loss:{:.3f}, momentum_loss: {:,.3f}, continuity_loss: {:,.3f}, lr = {:.6f}'.format( \
                    total_loss, multi_stft_loss, spec_loss, time_loss, energy_loss, momentum_loss, continuity_loss, self.optimizer.param_groups[0]['lr']))
            pbar.close()

            ## Train loss in a current epoch
            num_batches = len(self.train_loader)
            train_avg_losses = {key: train_losses[key] / num_batches for key in train_losses}

            ## scheduler step
            self.scheduler.step()
            self.prior_scheduler.step()

            ## Logging learning rate
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
                self.writer.add_scalar(f'learning rate', current_lr, self.current_epoch)

            ## save model
            if self.current_epoch % self.save_epochs == 0:
                ckptname = self.save_checkpoint()
                self.logger.info('Saved checkpoints at {}'.format(ckptname))

            ## test
            if self.current_epoch % self.val_epochs == 0:
                self.logger.info("Start evaluation")
                self.renderer.eval()
                valid_avg_losses, valid_avg_metrics = self.test() 

                ## Save best model
                if valid_avg_losses['total_loss'] < self.best_valid_loss:
                    self.best_valid_loss = valid_avg_losses['total_loss']
                    self.save_checkpoint(mode='best')
                    self.logger.info("Best valid loss updated")

                ## Logging
                self.log_tensorboard(losses=valid_avg_losses, metrics=valid_avg_metrics, cur_iter=self.current_epoch, mode_set="test")
                self.logger.info("Evaluations. Current Epoch:%d", self.current_epoch)
                self.logger.info("loss:{:.4f}, multi stft loss:{:.3f}, spec loss:{:.3f}, amplitude loss: {:.3f}, angle loss: {:.3f}, time loss:{:.3f}, energy loss:{:.3f}, momentum loss:{:.5f}, continuity loss:{:.5f}".format( \
                    valid_avg_losses['total_loss'], valid_avg_losses['multi_stft_loss'], valid_avg_losses['spec_loss'], valid_avg_losses['amplitude_loss'], valid_avg_losses['angle_loss'], valid_avg_losses['time_loss'], valid_avg_losses['energy_loss'], \
                    valid_avg_losses['momentum_loss'], valid_avg_losses['continuity_loss']))

                self.logger.info('Angle:{:.3f}, Amplitude:{:.4f}, Envelope:{:.4f}, T60:{:.5f}, C50:{:.5f}, EDT:{:.5f}, multi_stft:{:.4f}'.format( \
                valid_avg_metrics['Angle'], valid_avg_metrics['Amplitude'], valid_avg_metrics['Envelope'], valid_avg_metrics['T60'], valid_avg_metrics['C50'], valid_avg_metrics['EDT'], valid_avg_metrics['multi_stft']))
                self.renderer.train()

            if self.current_epoch % self.prune_epochs == 0 and self.current_epoch > self.prune_start_epoch:
                self.renderer.eval()
                with torch.no_grad():
                    self.logger.info("Start pruning")
                    ori_position_mps, new_position_mps, ori_s_t_energy, new_s_t_energy, energy_threshold = self.renderer_module.pruning()

                self.logger.info(f"Current multipoles count:{self.renderer_module.get_count_mps()}")

                self.update_optimizer()
                self.renderer.train()

            if self.current_epoch % self.val_epochs == 0:
                self.log_tensorboard_loss(train_losses=train_avg_losses, val_losses=valid_avg_losses, cur_epoch=self.current_epoch, mode_set="loss")
            else:
                self.log_tensorboard_train_loss(train_losses=train_avg_losses, cur_epoch=self.current_epoch, mode_set="loss")


    def update_optimizer(self):
        old_lr = self.optimizer.param_groups[0]['lr']
        old_weight_decay = self.optimizer.param_groups[0]['weight_decay']
        old_betas = self.optimizer.param_groups[0]['betas']
        
        self.optimizer = torch.optim.Adam(
            self.renderer_module.parameters(),
            lr=old_lr,
            weight_decay=old_weight_decay,
            betas=old_betas
         )

        old_scheduler_state_dict = self.scheduler.state_dict()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
            T_max=float(self.kwargs_train['T_max']), eta_min=float(self.kwargs_train['eta_min']),
            last_epoch=-1)
        self.scheduler.load_state_dict(old_scheduler_state_dict)


    def test(self):
        valid_losses = {'total_loss': 0, 'spec_loss': 0, 'amplitude_loss': 0, 'angle_loss': 0, 'time_loss': 0, 'energy_loss': 0, 'multi_stft_loss': 0, 'momentum_loss': 0, 'continuity_loss': 0, 'data_loss': 0, 'physical_loss': 0}
        valid_metrics = {'Angle': 0, 'Amplitude': 0, 'Envelope': 0, 'T60': 0, 'C50': 0, 'EDT': 0, 'multi_stft': 0}

        for check_idx, test_batch in enumerate(self.test_loader):
            ori_sig, position_rx, position_tx = test_batch
            with torch.no_grad():
                pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda())
                pred_sig = pred_sig[...,0] + 1j * pred_sig[...,1]
                ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)
            coords_pde = self.test_set._sample_collocation_points(self.test_set.n_collocation)
            coords_pde = torch.from_numpy(coords_pde).cuda()
            with torch.enable_grad():
                losses, metrics, ori_time, pred_time = self.calculate_metrics(coords_pde, position_tx.cuda(), pred_sig, ori_sig, self.fs)

            for key in valid_losses:
                valid_losses[key] += losses[key].detach()

            for key in valid_metrics:
                valid_metrics[key] += metrics[key]

        num_batches = len(self.test_loader) 
        valid_avg_losses = {key: valid_losses[key] / num_batches for key in valid_losses}
        valid_avg_metrics = {key: valid_metrics[key] / num_batches for key in valid_metrics}

        return valid_avg_losses, valid_avg_metrics

        
    def calculate_metrics(self, coords_pde, position_tx, pred_sig, ori_sig, fs):
        """ calculate the metrics and losses
        """
        # Loss calculation
        spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, momentum_loss, continuity_loss, ori_time, pred_time = self.criterion(
            self.renderer_module, coords_pde, position_tx.cuda(), pred_sig, ori_sig, create_graph=False)
        total_loss, data_loss, physics_loss = self.uncertainty_weighted_loss(
            spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, momentum_loss, continuity_loss)

        # Metrics calculation. metric_cal only handles [B, T], so the 4 FOA channels are
        # folded into the batch dimension
        B, T, C = ori_time.shape
        ori_flat = ori_time.permute(0, 2, 1).reshape(B * C, T)
        pred_flat = pred_time.permute(0, 2, 1).reshape(B * C, T)
        angle_error, amp_error, env_error, t60_error, edt_error, C50_error, multi_stft, _, _ = metric_cal(
            ori_flat.detach().cpu().numpy(),
            pred_flat.detach().cpu().numpy(),
            fs=fs
        )

        losses = {
            'total_loss': total_loss,
            'spec_loss' : spec_loss,
            'amplitude_loss': amplitude_loss,
            'angle_loss': angle_loss,
            'time_loss' : time_loss,
            'energy_loss': energy_loss,
            'multi_stft_loss': multi_stft_loss,
            'momentum_loss'  : momentum_loss,
            'continuity_loss': continuity_loss,
            'data_loss'      : data_loss,
            'physical_loss'  : physics_loss
        }

        metrics = {
            'Angle': angle_error,
            'Amplitude': amp_error,
            'Envelope': env_error,
            'T60': t60_error,
            'C50': C50_error,
            'EDT': edt_error,
            'multi_stft': multi_stft
        }
        return losses, metrics, ori_time, pred_time
    

    def log_tensorboard_loss(self, train_losses, val_losses=None, cur_epoch=None, mode_set='loss'):
        if cur_epoch % self.val_epochs == 0:
            for key in train_losses.keys():
                self.writer.add_scalars(
                    main_tag=f"{mode_set}/{key}",  # tag
                    tag_scalar_dict={
                        'train': train_losses[key],
                        'val': val_losses[key]
                    },
                    global_step=cur_epoch
                    )

    def log_tensorboard_train_loss(self, train_losses, cur_epoch=None, mode_set='loss'):
        # called on the epochs where no validation ran, so no `% val_epochs` guard here
        for key in train_losses.keys():
            self.writer.add_scalars(
                main_tag=f"{mode_set}/{key}",  # tag
                tag_scalar_dict={
                    'train': train_losses[key],
                },
                global_step=cur_epoch
                )

    def log_tensorboard(self, losses=None, metrics=None, cur_iter=None, mode_set="train"):
        for loss_name, value in losses.items():
            self.writer.add_scalar(f'{mode_set}_loss/{loss_name}', value, cur_iter)
    
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'{mode_set}_metric/{metric_name}', value, cur_iter)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/leejs/NAMS/config_files/pyc4.yml', help='config file path')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        kwargs = yaml.load(file, Loader=yaml.FullLoader)

    ## log
    logdir = os.path.join(kwargs['path']['logdir'])
    os.makedirs(logdir, exist_ok=True)

    # copy the config file
    dest_file_path = os.path.join(logdir, 'experiment_config.yml')
    copyfile(args.config, dest_file_path)

    worker = NAMS_Runner(**kwargs)
    worker.train()


# python nams_runner.py --config config_files/meshrir.yml 