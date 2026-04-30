import os
import argparse
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets_loader import WaveLoader
from model import NAMS_network
from renderer import NAMS_renderer
from utils.metric import metric_cal
from utils.logger import logger_config
from utils.criterion import Criterion

SEED = 2025

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)


class Tester():
    def __init__(self, model_name, ckptsdir, **kwargs) -> None:

        self.model_name = model_name

        kwargs_path = kwargs['path']
        kwargs_render = kwargs['render']
        kwargs_network = kwargs['model']
        kwargs_train = kwargs['train']

        ## Path settings
        self.dataset_type = kwargs_path['dataset_type']
        self.logdir = kwargs_path['logdir']
        self.ckptsdir = ckptsdir 

        ## Logger
        log_filename = "logger_test.log"
        log_savepath = os.path.join(self.logdir, log_filename)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='nams')
        self.logger.info("data type:%s, logdir:%s", self.dataset_type, self.logdir)

        # Train settings
        self.batch_size = 1
        self.current_epoch = 0
        self.devices = torch.device('cuda')

        if self.dataset_type == 'MeshRIR':
            self.ir_length = 0.1
            self.fs = 24000
            self.dataset_dir = '/dataset/meshrir/'
        elif self.dataset_type == 'Treble':
            self.ir_length = 0.1
            self.fs = 24000
            self.dataset_dir = '/dataset/apartment566/'
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_type}")

        self.train_set = WaveLoader(base_folder=self.dataset_dir, dataset_type=self.dataset_type, eval=False, seq_len=int(self.ir_length*self.fs), fs=self.fs)
        self.test_set = WaveLoader(base_folder=self.dataset_dir, dataset_type=self.dataset_type, eval=True, seq_len=int(self.ir_length*self.fs), fs=self.fs)
    
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.spk_pos = self.train_set.positions_tx[0]
        self.logger.info("Train set size:%d, Test set size:%d", len(self.train_set), len(self.test_set))


        ## network, renderer
        if self.model_name == 'NAMS':
            nams_network = NAMS_network(kwargs_network, kwargs_render) # network
            self.renderer = NAMS_renderer(networks_fn=nams_network, spk_pos=self.spk_pos, **kwargs_render) # renderer
        else:
            raise ValueError(f"Invalid model: {model_name}")

        # loss settings
        self.criterion = Criterion(kwargs_train)


    def load_checkpoints(self):
        if self.model_name == 'NAMS':
            ckptsdir = os.path.join(self.logdir, 'ckpts', self.ckptsdir)
            ckpt = torch.load(ckptsdir, map_location=self.devices)
            self.logger.info('Loading ckpts %s', ckptsdir)
            multipoles_count = ckpt['renderer_state_dict']['position_mps'].size(0)
            print(multipoles_count)
            self.renderer.multipoles_count_reset(multipoles_count)
            self.renderer.load_state_dict(ckpt['renderer_state_dict'])
            self.renderer = self.renderer.cuda()

        else:
            raise ValueError(f"Invalid model: {self.model}")


    def test(self):
        params = list(self.renderer.parameters())
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.logger.info("Total number of parameters: %s", total_params)

        self.logger.info("Parameters per layers: ")
        for name, param in self.renderer.named_parameters():
            self.logger.info(f"{name} : {param.numel()}")    

        if self.model_name == 'NAMS':
            self.logger.info(f"mps count: {self.renderer.get_count_mps()}")

        test_losses = {'total_loss': 0, 'spec_loss': 0, 'amplitude_loss': 0, 'angle_loss': 0, 'time_loss': 0, 'energy_loss': 0, 'multi_stft_loss': 0}
        test_metrics = {'Angle': 0, 'Amplitude': 0, 'Envelope': 0, 'T60': 0, 'C50': 0, 'EDT': 0, 'multi_stft': 0}

        self.renderer.eval()
        for check_idx, test_batch in enumerate(self.test_loader):
            with torch.no_grad():
                ori_sig, position_rx, position_tx = test_batch

                position_rx = position_rx.cuda()
                position_tx = position_tx.cuda()
                pred_sig = self.renderer(position_rx, position_tx)
                pred_sig = pred_sig[...,0] + 1j * pred_sig[...,1]
                ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                losses, metrics, ori_time, pred_time = self.calculate_metrics(pred_sig, ori_sig, self.fs)

            for key in test_losses:
                test_losses[key] += losses[key].detach()

            for key in test_metrics:
                test_metrics[key] += metrics[key]

        num_batches = len(self.test_loader) 
        test_avg_losses = {key: test_losses[key] / num_batches for key in test_losses}
        test_avg_metrics = {key: test_metrics[key] / num_batches for key in test_metrics}

        self.logger.info(f"Test avg losses: {test_avg_losses}")
        self.logger.info(f"Test avg metrics: {test_avg_metrics}")


        
    def calculate_metrics(self, pred_sig, ori_sig, fs):
        """ calculate the metrics and losses
        """
        # Loss calculation
        spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, ori_time, pred_time = self.criterion(pred_sig, ori_sig)
        total_loss = spec_loss + amplitude_loss + angle_loss + time_loss + energy_loss + multi_stft_loss
        # Metrics calculation
        angle_error, amp_error, env_error, t60_error, edt_error, C50_error, multi_stft, _, _ = metric_cal(
            ori_time.detach().cpu().numpy(), 
            pred_time.detach().cpu().numpy(), 
            fs=fs
        )

        losses = {
            'total_loss': total_loss,
            'spec_loss': spec_loss,
            'amplitude_loss': amplitude_loss,
            'angle_loss': angle_loss,
            'time_loss': time_loss,
            'energy_loss': energy_loss,
            'multi_stft_loss': multi_stft_loss
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='config file path')
    parser.add_argument('--model_name', type=str, default='NAMS', help='model name')
    parser.add_argument('--ckptsdir', type=str, default='best.tar', help='ckpts directory') 
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        kwargs = yaml.load(file, Loader=yaml.FullLoader)

    ## log
    logdir = os.path.join(kwargs['path']['logdir'])

    worker = Tester(model_name=args.model_name, ckptsdir=args.ckptsdir, **kwargs)
    worker.load_checkpoints()
    worker.test()

