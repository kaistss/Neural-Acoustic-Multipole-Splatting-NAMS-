import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np


class NAMS_network(nn.Module):
    def __init__(self, cfg, render_cfg):
        super().__init__()

        pos_encoding_sigma = cfg["pos_encoding_sigma"]
        dir_encoding_sig = cfg["dir_encoding_sig"]

        signal_network = cfg['signal_network']
        sh_network = cfg['sh_network']

        self.signal_output_dim = cfg['signal_output_dim']  # Time
        self.sh_output_dim = cfg['sh_output_dim'] # 1/3 oct
        self.sh_mode = cfg['sh_mode'] 
        self.sh_coeff_length = cfg['sh_coeff_length']

        self._pos_encoding = tcnn.Encoding(3, pos_encoding_sigma, dtype=torch.float32)
        self._dir_encoding = tcnn.Encoding(3, dir_encoding_sig, dtype=torch.float32)

        n_signal_input = self._pos_encoding.n_output_dims 
        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=signal_network,
        )

        n_sh_input = self._pos_encoding.n_output_dims  + self._dir_encoding.n_output_dims
        self._model_sh = tcnn.Network(
            n_input_dims=n_sh_input,
            n_output_dims=self.sh_output_dim,
            network_config=sh_network,
        )


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


