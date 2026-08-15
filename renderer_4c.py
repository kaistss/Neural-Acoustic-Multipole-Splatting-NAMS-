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
        self.early     = kwargs['early']
        self.fs = kwargs['fs']
        self.c = kwargs['speed']
        self.xyz_min = kwargs['xyz_min']
        self.xyz_max = kwargs['xyz_max']
        self.dirs_min = kwargs['dirs_min']
        self.dirs_max = kwargs['dirs_max']


        self.sphere_radius_interval = kwargs['init_sphere_radius_interval']
        self.sphere_points_per_shell = kwargs['init_sphere_points_per_shell']
        self.candidate_walls = kwargs['candidate_walls']
        self.prune_threshold = kwargs['prune_threshold']

        self.spk_pos = spk_pos
        self.xyz_trainable = kwargs['xyz_trainable']
        
        if self.xyz_trainable:
            self.position_mps   = nn.Parameter(torch.tensor(self.multipoles_init_spherical()).float(), requires_grad=True)
            self.position_walls = nn.Parameter(, requires_grad=True)
        else:
            self.position_mps = nn.Parameter(torch.tensor(self.multipoles_init_spherical()).float(), requires_grad=False)

    def imagesource_mirror(self, pos_spk):
        


    def forward(self, pos_rcv, pos_spk):

        batch_size = pos_rcv.size(0)

        position_mps = self.position_mps.unsqueeze(0).expand(batch_size, -1, -1) # [N, 3] -> [B, N, 3]

        position_rcv = pos_rcv.unsqueeze(1) # [B, 3] -> [B, 1, 3]
        position_spk = pos_spk.unsqueeze(1) # [B, 3] -> [B, 1, 3]

        dirs = position_mps - position_rcv # [B, N, 3]
        r = torch.norm(dirs, dim=-1).unsqueeze(-1) # [B, N, 1]
        dirs_norm = dirs / (dirs.norm(dim=-1).unsqueeze(-1) + EPS) # [B, N, 3]

        network_mps = normalize_points(position_mps, self.xyz_min, self.xyz_max) # [B, N, 3] # 0 ~ 1 사이 값으로 바뀌네
        network_dirs =  normalize_points(dirs, self.dirs_min, self.dirs_max) # [B, N, 3]

        s_t, sh = self.network_fn(network_mps, network_dirs) # [B, N, M, L] # N은 멀티폴 수, M은 Spherical Harmonics 기저 수

        n_pad = int(self.ir_length*1.5*self.fs)
        s_t_padded = torch.zeros(s_t.size(0), s_t.size(1), n_pad, device=s_t.device)
        s_t_padded[:, :, :s_t.size(-1)] = s_t.float() # [B, N, 1.5T]

        S_f_padded = torch.fft.rfft(s_t_padded, dim=-1) # [B, N, F]

        frequencies = torch.fft.rfftfreq(n_pad, d=1/self.fs, device=s_t.device).unsqueeze(0).unsqueeze(0) # [1, 1, F]

        dir_component = sh_expansion(3, sh.float(), dirs_norm, n_pad) # [B, N, F]
        x, y, z     = dirs_norm[..., 0], dirs_norm[..., 1], dirs_norm[..., 2]
        foa_gain = torch.stack([
            C0 * torch.ones_like(x),
            C1 * x,
            C1 * y,
            C1 * z,
        ], dim=-1) # [B, N, 4] 데이터로더에서 wzyx를 wxyz로 바꿔서 공급한다
        delay_time = r / self.c  # [B, N, 1]
        phase_shift = torch.exp(-1j * 2 * torch.pi * frequencies * delay_time)  # [B, N, F]
        attenuation = 1.0 / (r + 1e-6)  # [B, N, 1] 

        S_f_processed = S_f_padded * phase_shift * attenuation * dir_component # [B, N, F] complex
        # einsum instead of a [B, N, F, 4] intermediate before the sum over multipoles
        S_f_sum       = torch.einsum('bnf,bnc->bfc', S_f_processed, foa_gain.to(S_f_processed.dtype)) # [B, F, 4]
        s_t_sum = torch.fft.irfft(S_f_sum, dim = 1) # [B, T, 4])
        s_t_sum = s_t_sum[:, :int(self.ir_length*self.fs), :]

        # masking (from AVR) : option, it can be used to mask the RIR based on the distance between the receiver and the speaker
        r_rcv_spk = torch.norm(position_rcv - position_spk, dim=-1) # [B, N]
        sample_rcv_spk = torch.clamp(torch.round((r_rcv_spk / self.c - 0.001) * self.fs), min=0, max=s_t_sum.size(1) - 1) # 1ms margin
        range_tensor = torch.arange(s_t_sum.size(1), device=s_t.device) 
        zero_mask_rcv2mps = range_tensor >= sample_rcv_spk
        s_t_sum = s_t_sum * zero_mask_rcv2mps.unsqueeze(-1)
    
        H = torch.fft.rfft(s_t_sum, dim=1)  # [B, F, 4]
        H = torch.cat([torch.real(H).unsqueeze(-1), torch.imag(H).unsqueeze(-1)], dim=-1) # [B, F, 4, 2]
        
        return H
        
    # 이건 Forward와 다르게, t도메인 미분이 필요하므로 별도의 파이프라인을 사용함
    def evaluate_field(self, coords_xyzt, pos_src):
        """
        coords_xyzt: [N, 4] (x, y, z, t), requires_grad=True는 호출부(criterion)에서 설정됨
        pos_src:     [N, 3] 각 collocation point에 대응하는 source 위치 (현재는 미사용, 아래 설명 참고)

        Returns: [N, 4] = (W, X, Y, Z) — momentum/continuity 계산을 위한 (pressure, velocity_x/y/z)
        """
        pos_rcv = coords_xyzt[:, :3]     # [N, 3]
        t_query = coords_xyzt[:, 3]       # [N]
        N = pos_rcv.size(0)

        # --- 여기까지는 forward()와 동일한 로직, receiver를 collocation point 위치로 사용 ---
        position_mps = self.position_mps.unsqueeze(0).expand(N, -1, -1)   # [N, M, 3]
        position_rcv = pos_rcv.unsqueeze(1)                                 # [N, 1, 3]

        dirs = position_mps - position_rcv
        r = torch.norm(dirs, dim=-1).unsqueeze(-1)
        dirs_norm = dirs / (dirs.norm(dim=-1).unsqueeze(-1) + EPS)

        network_mps = normalize_points(position_mps, self.xyz_min, self.xyz_max)
        network_dirs = normalize_points(dirs, self.dirs_min, self.dirs_max)

        s_t, sh = self.network_fn(network_mps, network_dirs)

        n_pad = int(self.ir_length * 1.5 * self.fs)
        s_t_padded = torch.zeros(s_t.size(0), s_t.size(1), n_pad, device=s_t.device)
        s_t_padded[:, :, :s_t.size(-1)] = s_t.float()

        S_f_padded = torch.fft.rfft(s_t_padded, dim=-1)   # [N, M, F]
        frequencies = torch.fft.rfftfreq(n_pad, d=1/self.fs, device=s_t.device)   # [F]

        dir_component = sh_expansion(3, sh.float(), dirs_norm, n_pad)   # [N, M, F]

        x, y, z = dirs_norm[..., 0], dirs_norm[..., 1], dirs_norm[..., 2]
        foa_gain = torch.stack([
            C0 * torch.ones_like(x),
            C1 * x,
            C1 * y,
            C1 * z,
        ], dim=-1)   # [N, M, 4]

        delay_time = r / self.c
        phase_shift = torch.exp(-1j * 2 * torch.pi * frequencies.view(1,1,-1) * delay_time)
        attenuation = 1.0 / (r + 1e-6)

        S_f_processed = S_f_padded * phase_shift * attenuation * dir_component
        S_f_sum = torch.einsum('nmf,nmc->nfc', S_f_processed, foa_gain.to(S_f_processed.dtype))   # [N, F, 4]

        # --- 여기부터는 forward()와 다른 지점: irfft 대신 연속시간 합성 ---
        pred = fourier_synthesize(S_f_sum, frequencies, t_query, n_pad)   # [N, 4] = (W,X,Y,Z)

        return pred
    
    def multipoles_count_reset(self, multipoles_count):
        if self.xyz_trainable:
            self.position_mps = nn.Parameter(torch.zeros(multipoles_count, 3).float(), requires_grad=True)
        else:
            self.position_mps = nn.Parameter(torch.zeros(multipoles_count, 3).float(), requires_grad=False)

    def wall_init_spherical(self):
        center = self.spk_pos.detach().numpy()
        max_radius = self.early * self.c / 2 # 대충 1차 반사 벽이 존재할 수 있는 거리: 0.06초 이내에 벽 찍고 왕복이 가능해야 하므로
        candidate_walls_array = np.full(1, self.candidate_walls)
        sphere_points = fibonacci_sphere_sampling(candidate_walls_array, max_radius)
        sphere_points = transform.Rotation.random().apply(sphere_points) # 벽의 위치와 노말벡터, 총 2가지의 현실 데이터를 시뮬레이션해야 함
        sphere_points = sphere_points + center
        
        

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


def sh_expansion(deg, sh, dirs, n_pad):
    """ sh: [B, N, M, L] time-domain SH coefficients, dirs: [B, N, 3] unit vectors.

    The SH weights are real and frequency independent, so weighting in the time domain
    and then taking the rfft is identical to the frequency-domain version, while never
    materializing the [B, N, M, F] spectrum (16x less memory).
    """
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

    shs = (sh_basis * sh).sum(dim=-2)                 # [B, N, L]
    shs = torch.fft.rfft(shs, n=n_pad, dim=-1)        # [B, N, F]
    shs = shs / (shs.norm(dim=-1).unsqueeze(-1) + EPS)

    return shs

def fourier_synthesize(S_f, freqs, t, n_pad):
    F_bins = freqs.shape[0]
    weight = torch.full((F_bins,), 2.0, device=S_f.device, dtype=S_f.real.dtype)
    weight[0] = 1.0
    if n_pad % 2 == 0:
        weight[-1] = 1.0
    phase = 2 * torch.pi * freqs.unsqueeze(0) * t.unsqueeze(-1)
    complex_exp = torch.exp(1j * phase)
    terms = S_f * complex_exp.unsqueeze(-1) * weight.view(1, -1, 1) # [N, F]
    p = torch.real(torch.sum(terms,dim=1)) / n_pad
    return p