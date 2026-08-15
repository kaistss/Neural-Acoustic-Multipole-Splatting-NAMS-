import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FrequencyEncoding(nn.Module):
    def __init__(self, n_input_dims = 3, n_frequencies = 10):
        super().__init__()
        self.n_input_dims  = n_input_dims
        self.n_frequencies = n_frequencies
        self.n_output_dims = n_input_dims * n_frequencies * 2

        # 2^0, 2^1, ... 을 미리 계산해서 버퍼에 등록
        freq_bands         = 2.0 ** torch.arange(n_frequencies)
        self.register_buffer('freq_bands', freq_bands) 

    def forward(self, x):
        x_scaled = x.unsqueeze(-1) * self.freq_bands * torch.pi
        sin_part = torch.sin(x_scaled) # [..., n_input_dims, L]
        cos_part = torch.cos(x_scaled) # [..., n_input_dims, L]

        out      = torch.stack([sin_part, cos_part], dim = -1)
        out      = out.reshape(*x.shape[:-1], self.n_output_dims)
        return out

class NAMS_network(nn.Module):
    def __init__(self, cfg, render_cfg):
        super().__init__()

        pos_encoding_sigma = cfg["pos_encoding_sigma"]
        dir_encoding_sigma = cfg["dir_encoding_sig"]

        signal_network = cfg['signal_network']
        sh_network = cfg['sh_network']

        self.signal_output_dim = cfg['signal_output_dim']  # Time
        self.sh_output_dim = cfg['sh_output_dim'] # 1/3 oct
        self.sh_mode = cfg['sh_mode'] 
        self.sh_coeff_length = cfg['sh_coeff_length']

        self._pos_encoding = FrequencyEncoding(n_input_dims=3, n_frequencies=pos_encoding_sigma['n_frequencies'])
        self._dir_encoding = FrequencyEncoding(n_input_dims=3, n_frequencies=dir_encoding_sigma['n_frequencies'])

        n_signal_input = self._pos_encoding.n_output_dims 
        self._model_signal = nn.Sequential()
        for i in range(signal_network['n_hidden_layers']):
            if i == 0:
                self._model_signal.append(nn.Linear(n_signal_input, signal_network['n_neurons']))
                self._model_signal.append(nn.ReLU())
            else:
                self._model_signal.append(nn.Linear(signal_network['n_neurons'], signal_network['n_neurons']))
                self._model_signal.append(nn.ReLU())
        self._model_signal.append(nn.Linear(signal_network['n_neurons'], self.signal_output_dim))
        n_sh_input = self._pos_encoding.n_output_dims  + self._dir_encoding.n_output_dims
        self._model_sh     = nn.Sequential()
        for i in range(sh_network['n_hidden_layers']):
            if i == 0:
                self._model_sh.append(nn.Linear(n_sh_input, sh_network['n_neurons']))
                self._model_sh.append(nn.ReLU())
                
            else:
                self._model_sh.append(nn.Linear(sh_network['n_neurons'], sh_network['n_neurons']))
                self._model_sh.append(nn.ReLU())
                
        self._model_sh.append(nn.Linear(sh_network['n_neurons'], self.sh_output_dim))


    def forward(self, position_mps, dirs):
        mps_size = position_mps.size(1)

        position_mps_flat = position_mps.reshape(-1, 3) # [B, N, 3] -> [B*N, 3]
        dirs_flat = dirs.reshape(-1, 3) # [B, N, 3] -> [B*N, 3]

        pos_embedding = self._pos_encoding(position_mps_flat) # [B*N, 3] -> [B*N, E]
        dir_embedding = self._dir_encoding(dirs_flat) # [B*N, 3] -> [B*N, E]

        # feature_signal = pos_embedding
        signal = self._model_signal(pos_embedding).reshape(-1, mps_size, self.signal_output_dim) #[B*N, T] -> [B, N, T]

        feature_sh = torch.cat([pos_embedding, dir_embedding], -1) # [B*N, ]

        sh = self._model_sh(feature_sh).reshape(-1, mps_size, self.sh_mode, self.sh_coeff_length) #[B*N, E] -> [B, N, M, L]

        return signal, sh


