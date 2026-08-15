import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import re
import librosa
from pydoe import lhs

class WaveLoader(Dataset):
    def __init__(self, base_folder, dataset_type='MeshRIR', eval=False, seq_len=2048, fs=16000, n_collocation = 32):

        self.wave_chunks = []
        self.positions_rx = []
        self.positions_tx = []
        self.rotations_tx = []
        self.wave_max = float('-inf')
        self.wave_min = float('inf')
        self.position_max = np.array([float('-inf'), float('-inf'), float('-inf')])
        self.position_min = np.array([float('inf'), float('inf'), float('inf')])
        # IR length in seconds; also the upper bound of the collocation points' time axis.
        # Derived from the caller's seq_len/fs so it cannot drift from the config.
        self.ir_length = seq_len / fs
        self.dataset_type = dataset_type
        self.eval = eval

        # load three different datasets seperately
        if dataset_type == 'MeshRIR':
            self.load_mesh_rir(base_folder, eval, seq_len, fs)
        elif dataset_type == 'Treble':
            self.load_simu(base_folder, eval, seq_len, fs)
        elif dataset_type == 'Pyroomacoustics':
            self.load_pyroomacoustics(base_folder, eval, seq_len, fs)
        else:
            raise ValueError("Unsupported dataset type")

        self.wave_chunks = torch.tensor(np.array(self.wave_chunks), dtype=torch.complex64)
        self.positions_rx = torch.tensor(np.array(self.positions_rx), dtype=torch.float32)
        self.positions_tx = torch.tensor(np.array(self.positions_tx), dtype=torch.float32)
        self.n_collocation = n_collocation


    def compute_bounds(self, omega_center, omega_size):
        half = omega_size / 2.0
        lb = torch.tensor([1.0 * (omega_center[0] - half), 1.0 * (omega_center[1] - half),
                           1.0 * (omega_center[2] - half)], dtype=torch.float32)
        ub = torch.tensor([1.0 * (omega_center[0] + half), 1.0 * (omega_center[1] + half),
                           1.0 * (omega_center[2] + half)], dtype=torch.float32)
        return lb.numpy(), ub.numpy()

    def _room_prefix(self, roomnum):
        return f"R{roomnum:02d}"

    def _sample_collocation_points(self, n_collocation):
        unit_samples = lhs(4, samples=n_collocation)
        lb_np = np.concatenate([self.position_min, [0.0]])
        ub_np = np.concatenate([self.position_max, [self.ir_length]])
        return (unit_samples * (ub_np - lb_np) + lb_np).astype(np.float32)

    def load_pyroomacoustics(self, base_folder, eval, seq_len, fs=8000):

        seed  = 2026
        status_ok = "Success"
        csv_path = os.path.join(base_folder, "Generated_HOA_SRIR_data.csv")
        data_dir = base_folder
        metadata = pd.read_csv(csv_path)
        prefix = self._room_prefix(0) # 방 번호가 0번이라서 0 넣음
        room_mask = metadata['Name'].astype(str).str.match(rf'^{re.escape(prefix)}[_\-]?')
        if status_ok is not None and 'status' in metadata.columns:
            room_mask &= (metadata['status'] == status_ok)
        room_meta = metadata[room_mask].reset_index(drop=True)
        if len(room_meta) == 0:
            raise ValueError(f"no rows found for room '{prefix}' (status='{status_ok}') in {csv_path}")
        x   = room_meta['x_mic'].to_numpy(dtype=np.float32)
        y   = room_meta['y_mic'].to_numpy(dtype=np.float32)
        z   = room_meta['z_mic'].to_numpy(dtype=np.float32)
        xs  = room_meta['x_source'].to_numpy(dtype=np.float32)
        ys  = room_meta['y_source'].to_numpy(dtype=np.float32)
        zs  = room_meta['z_source'].to_numpy(dtype=np.float32)

        foa_list = []
        fs = None
        n_samples = None
        SN3D_MISSING_GAIN = np.sqrt(3.0)
        RAW_TO_WXYZ       = [0, 3, 1, 2]
        for name in room_meta['Name']:
            wav_path = os.path.join(data_dir, name)
            audio, file_fs = librosa.load(wav_path, sr=None, mono=False)
            if audio.ndim != 2 or audio.shape[0] != 4:
                raise ValueError(f"{wav_path}: expected 4-channel FOA audio, got shape {audio.shape}")
            if fs is None:
                fs = file_fs
                n_samples = min(int(round(self.ir_length * fs)), audio.shape[1])
            elif file_fs != fs:
                raise ValueError(f"sample rate mismatch in {wav_path}: {file_fs} != {fs}")
            audio = audio.copy()
            audio[1:4, :] /= SN3D_MISSING_GAIN
            audio = audio[RAW_TO_WXYZ, :]
            foa_list.append(audio[:, :n_samples].T)  # (n_samples, 4)

        foa = np.stack(foa_list, axis=0).astype(np.float32)
        # 여기까지가 9261개 모두 받아오는 코드이고, 이걸 train test split 해야 함
        D_total      = len(x)
        rng  = np.random.default_rng(seed)
        perm = rng.permutation(D_total)
        n_train = 500
        n_val   = 300
        val_idx = perm[:n_val]
        train_idx = perm[n_val:n_val + n_train]
        if eval:
            foa = foa[val_idx]
            x, y, z = x[val_idx], y[val_idx], z[val_idx]
            xs, ys, zs = xs[val_idx], ys[val_idx], zs[val_idx]
        else:
            foa = foa[train_idx]
            x, y, z = x[train_idx], y[train_idx], z[train_idx]
            xs, ys, zs = xs[train_idx], ys[train_idx], zs[train_idx]

        wave_max     = np.max(foa).astype(np.float32)
        wave_min     = np.min(foa).astype(np.float32)
        center       = np.array([x.mean(), y.mean(), z.mean()])
        position_max = center + 0.5
        position_min = center - 0.5
        positions_rx = list(np.stack([x, y, z], axis = -1))
        positions_tx = list(np.stack([xs, ys, zs], axis = -1))
        # 마지막으로 wave_chunks 만들기
        #이건 1채널용임
        #wchannel = foa[:,:,0]
        #wave_data = np.fft.rfft(wchannel)
        wave_data = np.fft.rfft(foa, axis = 1) # [D, F, 4]
        self.wave_max = wave_max
        self.wave_min = wave_min
        self.position_max = position_max
        self.position_min = position_min
        self.positions_rx = positions_rx
        self.positions_tx = positions_tx
        self.wave_chunks  = wave_data
                


    def load_mesh_rir(self, base_folder, eval, seq_len, fs=24000):
        """ Load MeshRIR datasets
        """
        down_sample_rate = 48000 // fs

        if eval:
            wave_folder = os.path.join(base_folder, 'test')
        else:
            wave_folder = os.path.join(base_folder, 'train')

        filenames = [f for f in os.listdir(wave_folder) if f.endswith('.npy')]
        filenames.sort()

        rx_pos = np.load(os.path.join(base_folder, 'pos_mic.npy'))
        tx_pos = np.load(os.path.join(base_folder, 'pos_src.npy'))[0]

        for filename in filenames:
            file_ind = int(filename.split('_')[1].split('.')[0])
            position_rx = rx_pos[file_ind]
            position_tx = tx_pos
            audio_data = np.load(os.path.join(wave_folder, filename))[0,::down_sample_rate] # first resample the IR data
            
            st_idx = self.get_st_idx(audio_data, position_rx, position_tx, fs)
            
            wave_data = audio_data[st_idx:st_idx+seq_len] # index the IR data.
            wave_data = np.fft.rfft(wave_data)

            self.update_min_max(audio_data, position_rx)

            self.wave_chunks.append(wave_data)
            self.positions_rx.append(position_rx)
            self.positions_tx.append(position_tx)

    def get_st_idx(self, audio_data, position_rx, position_tx, fs):
        r = np.linalg.norm(position_rx - position_tx)
        onset = self.get_onset(audio_data)
        return onset - int(r * fs / 343)

    def get_onset(self, audio_data):
        x_abs = np.abs(audio_data)
        x_max = x_abs.max()
        indices = np.where(np.abs(audio_data) > 0.1 * x_max)[0]
        onset = indices[0]
        return onset
        

    def load_simu(self, base_folder, eval, seq_len, fs):
        """ Load Treble datasets
        """

        if eval:
            wave_folder = os.path.join(base_folder, 'test')
        else:
            wave_folder = os.path.join(base_folder, 'train')

        filenames = [f for f in os.listdir(wave_folder) if f.endswith('.npy')]
        filenames.sort()

        audio_data = np.load(os.path.join(wave_folder, 'irs.npy'))
        position_rx = np.load(os.path.join(wave_folder, 'rcv_pos.npy'))
        position_tx = np.load(os.path.join(wave_folder, 'spk_pos.npy'))
        position_tx = np.tile(position_tx, (len(position_rx), 1))

        audio_data = audio_data[:, :seq_len]
        wave_data = np.fft.rfft(audio_data)

        self.update_min_max(audio_data, position_rx)
        self.wave_chunks = wave_data
        self.positions_rx = position_rx
        self.positions_tx = position_tx

    def update_min_max(self, audio_data, position_rx):
        self.wave_max = max(self.wave_max, audio_data.max())
        self.wave_min = min(self.wave_min, audio_data.min())
        self.position_max = np.maximum(self.position_max, position_rx)
        self.position_min = np.minimum(self.position_min, position_rx)


    def __len__(self):
        return len(self.wave_chunks)

    def __getitem__(self, idx):
        wave_signal = self.wave_chunks[idx]
        position_rx = self.positions_rx[idx]
        position_tx = self.positions_tx[idx]
        
        return wave_signal, position_rx, position_tx

