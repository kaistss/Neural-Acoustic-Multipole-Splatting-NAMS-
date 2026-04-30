import torch
import numpy as np
import torch.nn as nn
import scipy.spatial.transform as transform


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    1.0925484305920792,
    0.31539156525252005,
    1.0925484305920792,
    0.5462742152960396
]
C3 = [
    0.5900435899266435,
    2.890611442640554,
    0.4570457994644658,
    0.3731763325901154,
    0.4570457994644658,
    1.445305721320277,
    0.5900435899266435
]
EPS = 1e-6


class NAMS_renderer(nn.Module):
    """ Audio signal rendering method
    """  
    def __init__(self, networks_fn, spk_pos, **kwargs) -> None:
        super().__init__()
        
        self.network_fn = networks_fn
        self.ir_length = kwargs['ir_length']
        self.fs = kwargs['fs']
        self.c = kwargs['speed']
        self.xyz_min = kwargs['xyz_min']
        self.xyz_max = kwargs['xyz_max']
        self.dirs_min = kwargs['dirs_min']
        self.dirs_max = kwargs['dirs_max']

        self.sphere_radius_interval = kwargs['init_sphere_radius_interval']
        self.sphere_points_per_shell = kwargs['init_sphere_points_per_shell']
        self.prune_threshold = kwargs['prune_threshold']

        self.spk_pos = spk_pos
        self.xyz_trainable = kwargs['xyz_trainable']
        
        if self.xyz_trainable:
            self.position_mps = nn.Parameter(torch.tensor(self.multipoles_init_spherical()).float(), requires_grad=True)
        else:
            self.position_mps = nn.Parameter(torch.tensor(self.multipoles_init_spherical()).float(), requires_grad=False)

    def forward(self, pos_rcv, pos_spk):

        batch_size = pos_rcv.size(0)

        position_mps = self.position_mps.unsqueeze(0).expand(batch_size, -1, -1) # [N, 3] -> [B, N, 3]
        position_rcv = pos_rcv.unsqueeze(1) # [B, 3] -> [B, 1, 3]
        position_spk = pos_spk.unsqueeze(1) # [B, 3] -> [B, 1, 3]

        dirs = position_mps - position_rcv # [B, N, 3]
        r = torch.norm(dirs, dim=-1).unsqueeze(-1) # [B, N, 1]
        dirs_norm = dirs / (dirs.norm(dim=-1).unsqueeze(-1) + EPS) # [B, N, 3]

        network_mps = normalize_points(position_mps, self.xyz_min, self.xyz_max) # [B, N, 3]
        network_dirs =  normalize_points(dirs, self.dirs_min, self.dirs_max) # [B, N, 3]

        s_t, sh = self.network_fn(network_mps, network_dirs) # [B, N, N, L]

        s_t_padded = torch.zeros(s_t.size(0), s_t.size(1), int(self.ir_length*1.5*self.fs), device=s_t.device)
        s_t_padded[:, :, :s_t.size(-1)] = s_t.float() # [B, N, 1.5T]

        S_f_padded = torch.fft.rfft(s_t_padded, dim=-1) # [B, N, F]

        frequencies = torch.fft.rfftfreq(s_t_padded.size(-1), d=1/self.fs, device=s_t.device).unsqueeze(0).unsqueeze(0) # [1, 1, F]

        sh_f_padded = torch.fft.rfft(sh.float(), n=int(self.ir_length*1.5*self.fs), dim=-1)
        dir_component = sh_expansion(3, sh_f_padded, dirs_norm) # / (Sh_f_padded.size(1)*16)

        delay_time = r / self.c  # [B, N, 1]
        phase_shift = torch.exp(-1j * 2 * torch.pi * frequencies * delay_time)  # [B, N, F]
        attenuation = 1.0 / (r + 1e-6)  # [B, N, 1] 

        S_f_processed = S_f_padded * phase_shift * attenuation * dir_component # [B, N, F] complex
        
        # sum
        S_f_sum = torch.sum(S_f_processed, dim=1) # [B, F] complex
        s_t_sum = torch.fft.irfft(S_f_sum, dim=-1) # [B, T])
        s_t_sum = s_t_sum[:, :int(self.ir_length*self.fs)]

        # masking (from AVR) : option, it can be used to mask the RIR based on the distance between the receiver and the speaker
        r_rcv_spk = torch.norm(position_rcv - position_spk, dim=-1) # [B, N]
        sample_rcv_spk = torch.clamp(torch.round((r_rcv_spk / self.c - 0.001) * self.fs), min=0, max=s_t_sum.size(-1) - 1) # 1ms margin
        range_tensor = torch.arange(s_t_sum.size(-1), device=s_t.device) 
        zero_mask_rcv2mps = range_tensor >= sample_rcv_spk
        s_t_sum = s_t_sum * zero_mask_rcv2mps 
    
        H = torch.fft.rfft(s_t_sum, dim=-1)  # [B, F]
        H = torch.cat([torch.real(H).unsqueeze(-1), torch.imag(H).unsqueeze(-1)], dim=-1) # [B, F, 2]
        
        return H
        
        
    def multipoles_count_reset(self, multipoles_count):
        if self.xyz_trainable:
            self.position_mps = nn.Parameter(torch.zeros(multipoles_count, 3).float(), requires_grad=True)
        else:
            self.position_mps = nn.Parameter(torch.zeros(multipoles_count, 3).float(), requires_grad=False)

    
    def multipoles_init_spherical(self):
        center = self.spk_pos.detach().numpy()

        max_radius = self.ir_length * self.c 

        radius_array = np.arange(self.sphere_radius_interval, max_radius, self.sphere_radius_interval)
        sphere_sampling_array = np.full(len(radius_array), self.sphere_points_per_shell)

        multipoles = [np.array([[0, 0, 0]])]
        for i, radius in enumerate(radius_array):
            sphere_points = fibonacci_sphere_sampling(sphere_sampling_array[i], radius)
            sphere_points = transform.Rotation.random().apply(sphere_points)
            multipoles.append(sphere_points)
        
        multipoles = np.vstack(multipoles)
        multipoles = multipoles + center 
        print(f'spherical multipoles.shape: {multipoles.shape}')        
        return multipoles
    
    def pruning(self):
        ori_position_mps = self.position_mps.clone()
        
        network_mps = normalize_points(ori_position_mps.unsqueeze(0), self.xyz_min, self.xyz_max)  # [1, N, 3]
        dummy_dirs = torch.zeros(1, ori_position_mps.size(0), 3, device=ori_position_mps.device) # [1, N, 3]
        
        ori_s_t, _ = self.network_fn(network_mps, dummy_dirs) # [1, N, L]
        ori_s_t_energy = torch.sum(ori_s_t ** 2, dim=-1).squeeze(0)  # [N]
        
        energy_threshold = torch.median(ori_s_t_energy) * self.prune_threshold
        keep_mask = ori_s_t_energy > energy_threshold
        new_s_t_energy = ori_s_t_energy[keep_mask]

        new_position_mps = ori_position_mps[keep_mask]

        if self.xyz_trainable:
            self.position_mps = nn.Parameter(new_position_mps, requires_grad=True)
        else:
            self.position_mps = nn.Parameter(new_position_mps, requires_grad=False)
        
        print(self.position_mps.device)
        return ori_position_mps, new_position_mps, ori_s_t_energy, new_s_t_energy, energy_threshold

    def get_position_mps(self):
        return self.position_mps.data.detach().cpu().numpy()

    def get_count_mps(self):
        return self.position_mps.size(-2)


def normalize_points(input_pts, xyz_min, xyz_max):
    return (input_pts - xyz_min) / (xyz_max - xyz_min)

def denormalize_points(input_pts, xyz_min, xyz_max):
    return input_pts * (xyz_max - xyz_min) + xyz_min

def fibonacci_sphere_sampling(num_points, radius=1.0):
    phi = np.pi * (3 - np.sqrt(5))
    indices = np.arange(num_points)

    y = 1 - (indices / (num_points - 1)) * 2
    unit_radius = np.sqrt(1 - y * y)
    theta = phi * indices
 
    x = np.cos(theta) * unit_radius * radius
    z = np.sin(theta) * unit_radius * radius
    y = y * radius
    
    return np.column_stack([x, y, z])


def sh_expansion(deg, sh, dirs):
    x, y, z = dirs[..., 0].unsqueeze(-1), dirs[..., 1].unsqueeze(-1), dirs[..., 2].unsqueeze(-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, yz, xz = x*y, y*z, x*z

    C0_t = torch.tensor(C0, device=dirs.device, dtype=dirs.dtype)
    C1_t = torch.tensor(C1, device=dirs.device, dtype=dirs.dtype)
    C2_t = torch.tensor(C2, device=dirs.device, dtype=dirs.dtype)
    C3_t = torch.tensor(C3, device=dirs.device, dtype=dirs.dtype)

    sh_basis = torch.stack([
        C0_t.repeat(sh.shape[0], sh.shape[1], 1),

        C1_t * y,
        C1_t * z, 
        C1_t * x,

        C2_t[0] * xy,
        C2_t[1] * yz,
        C2_t[2] * (2.0 * zz - xx - yy), # = 3zz - 1
        C2_t[3] * xz,
        C2_t[4] * (xx - yy),

        C3_t[0] * y * (3 * xx - yy),
        C3_t[1] * xy * z,
        C3_t[2] * y * (4 * zz - xx - yy),
        C3_t[3] * z * (2 * zz - 3 * xx - 3 * yy),
        C3_t[4] * x * (4 * zz - xx - yy),
        C3_t[5] * z * (xx - yy),
        C3_t[6] * x * (xx - 3 * yy)
    ]).permute(1, 2, 0, 3)

    shs = (sh_basis * sh).sum(dim=-2)
    shs = shs / (shs.norm(dim=-1).unsqueeze(-1))

    return shs

