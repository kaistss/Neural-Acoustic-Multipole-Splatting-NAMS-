"""
Loss (criterion) utilities in this module are adapted from the Acoustic Volume Rendering (AVR) codebase.

- Repository: https://github.com/penn-waves-lab/AVR
- Reference: Lan, Zitong, et al. "Acoustic volume rendering for neural impulse response fields."
  Advances in Neural Information Processing Systems 37 (2024): 44600-44623.
"""
import torch
import torch.nn as nn
import auraloss
import torch.nn.functional as F

class Criterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.spec_loss_weight = cfg['spec_loss_weight']
        self.time_loss_weight = cfg['time_loss_weight']
        self.energy_loss_weight = cfg['energy_loss_weight']
        self.multi_stft_weight = cfg['multistft_loss_weight']
        self.amplitude_loss_weight = cfg['amplitude_loss_weight']
        self.angle_loss_weight = cfg['angle_loss_weight']
    
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.mrft_loss = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1, fft_sizes=[512, 256, 128, 64], win_lengths=[300, 150, 75, 30], hop_sizes=[60, 30, 8, 4])

    def _grad(self, output_scalar, coords, create_graph=True):
        return torch.autograd.grad(
            output_scalar, coords,
            grad_outputs=torch.ones_like(output_scalar),
            create_graph=create_graph, retain_graph=True
        )[0]

    def physics_losses(self, model, coords_pde, coords_src, c0=343., loss_fn = F.l1_loss, create_graph = True):
        coords_pde = coords_pde.clone().requires_grad_(True)
        pred       = model.evaluate_field(coords_pde, coords_src)
        w, vx, vy, vz = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        dw = self._grad(w, coords_pde, create_graph)
        dvx = self._grad(vx, coords_pde, create_graph)
        dvy = self._grad(vy, coords_pde, create_graph)
        dvz = self._grad(vz, coords_pde, create_graph)
        momentum_residual = torch.stack([
            dw[:, 0] - (1. / c0) * dvx[:, 3],
            dw[:, 1] - (1. / c0) * dvy[:, 3],
            dw[:, 2] - (1. / c0) * dvz[:, 3],
        ], dim = -1)
        continuity_residual = (dvx[:, 0] + dvy[:, 1] + dvz[:, 2]) - (1. / c0) * dw[:, 3]

        l_momentum = loss_fn(momentum_residual, torch.zeros_like(momentum_residual))
        l_continuity = loss_fn(continuity_residual, torch.zeros_like(continuity_residual))
        return l_momentum, l_continuity


    
    def forward(self, model, coords_pde, coords_src, pred_sig, ori_sig, c0=343., create_graph=True):

        pred_time = torch.real(torch.fft.irfft(pred_sig, dim=1)) # [B, T, 4] 
        ori_time = torch.real(torch.fft.irfft(ori_sig, dim=1))   # [B, T, 4]
        B, T, C  = pred_time.shape

        pred_time_flat = pred_time.permute(0, 2, 1).reshape(B*C, T)
        ori_time_flat  = ori_time.permute(0, 2, 1).reshape(B*C, T)

        pred_spec = torch.abs(torch.stft(pred_time_flat, n_fft=256, return_complex=True))
        ori_spec = torch.abs(torch.stft(ori_time_flat, n_fft=256, return_complex=True))

        pred_spec_energy = torch.sum(pred_spec ** 2, dim=1) # [B * 4, frame]
        ori_spec_energy = torch.sum(ori_spec ** 2, dim=1)   # [B * 4, frame]

        predict_energy = torch.log10(torch.flip(torch.cumsum(torch.flip(pred_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        predict_energy -= predict_energy[:,[0]]
        ori_energy = torch.log10(torch.flip(torch.cumsum(torch.flip(ori_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        ori_energy -= ori_energy[:,[0]]

        real_loss = self.l1_loss(torch.real(pred_sig), torch.real(ori_sig))
        imag_loss  = self.l1_loss(torch.imag(pred_sig), torch.imag(ori_sig)) 
        spec_loss = (real_loss + imag_loss) * self.spec_loss_weight

        amplitude_loss = self.l1_loss(torch.abs(pred_sig), torch.abs(ori_sig)) * self.amplitude_loss_weight
        angle_loss = (self.l1_loss(torch.cos(torch.angle(pred_sig)), torch.cos(torch.angle(ori_sig)))+self.l1_loss(torch.sin(torch.angle(pred_sig)), torch.sin(torch.angle(ori_sig))))
        
        time_loss = self.l1_loss(ori_time, pred_time) * self.time_loss_weight
        energy_loss = self.l1_loss(ori_energy, predict_energy) * self.energy_loss_weight
        # auraloss expects [B, C, T]; channels are folded into the batch dim
        multi_stft_loss = self.mrft_loss(ori_time_flat.unsqueeze(1), pred_time_flat.unsqueeze(1)) * self.multi_stft_weight

        # 여기부터 Physics-Informed Loss
        # pred_time, ori_time 이렇게 2개 쓰면 될듯?
        momentum_loss, continuity_loss = self.physics_losses(model, coords_pde, coords_src, c0=c0, create_graph=create_graph)


        return spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, momentum_loss, continuity_loss, ori_time, pred_time