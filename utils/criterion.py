"""
Loss (criterion) utilities in this module are adapted from the Acoustic Volume Rendering (AVR) codebase.

- Repository: https://github.com/penn-waves-lab/AVR
- Reference: Lan, Zitong, et al. "Acoustic volume rendering for neural impulse response fields."
  Advances in Neural Information Processing Systems 37 (2024): 44600-44623.
"""
import torch
import torch.nn as nn
import auraloss

class Criterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.spec_loss_weight = cfg['spec_loss_weight']
        self.amplitude_loss_weight = cfg['amplitude_loss_weight']
        self.angle_loss_weight = cfg['angle_loss_weight']
        self.time_loss_weight = cfg['time_loss_weight']
        self.energy_loss_weight = cfg['energy_loss_weight']
        self.multi_stft_weight = cfg['multistft_loss_weight']
    
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.mrft_loss = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1, fft_sizes=[512, 256, 128, 64], win_lengths=[300, 150, 75, 30], hop_sizes=[60, 30, 8, 4])

    def forward(self, pred_sig, ori_sig):

        pred_time = torch.real(torch.fft.irfft(pred_sig, dim=-1)) 
        ori_time = torch.real(torch.fft.irfft(ori_sig, dim=-1))

        pred_spec = torch.abs(torch.stft(pred_time, n_fft=256, return_complex=True))
        ori_spec = torch.abs(torch.stft(ori_time, n_fft=256, return_complex=True))

        pred_spec_energy = torch.sum(pred_spec ** 2, dim=1)
        ori_spec_energy = torch.sum(ori_spec ** 2, dim=1)

        predict_energy = torch.log10(torch.flip(torch.cumsum(torch.flip(pred_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        predict_energy -= predict_energy[:,[0]]
        ori_energy = torch.log10(torch.flip(torch.cumsum(torch.flip(ori_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        ori_energy -= ori_energy[:,[0]]

        real_loss = self.l1_loss(torch.real(pred_sig), torch.real(ori_sig))
        imag_loss  = self.l1_loss(torch.imag(pred_sig), torch.imag(ori_sig)) 
        spec_loss = (real_loss + imag_loss) * self.spec_loss_weight

        amplitude_loss = self.l1_loss(torch.abs(pred_sig), torch.abs(ori_sig)) * self.amplitude_loss_weight 

        angle_loss = (self.l1_loss(torch.cos(torch.angle(pred_sig)), torch.cos(torch.angle(ori_sig))) + \
                    self.l1_loss(torch.sin(torch.angle(pred_sig)), torch.sin(torch.angle(ori_sig)))) * self.angle_loss_weight
        
        time_loss = self.l1_loss(ori_time, pred_time) * self.time_loss_weight

        energy_loss = self.l1_loss(ori_energy, predict_energy) * self.energy_loss_weight

        multi_stft_loss = self.mrft_loss(ori_time.unsqueeze(1), pred_time.unsqueeze(1)) * self.multi_stft_weight

        return spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, ori_time, pred_time