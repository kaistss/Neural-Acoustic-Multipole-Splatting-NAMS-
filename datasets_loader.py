import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WaveLoader(Dataset):
    def __init__(self, base_folder, dataset_type='MeshRIR', eval=False, seq_len=2048, fs=16000):

        self.wave_chunks = []
        self.positions_rx = []
        self.positions_tx = []
        self.rotations_tx = []

        self.wave_max = float('-inf')
        self.wave_min = float('inf')
        self.position_max = np.array([float('-inf'), float('-inf'), float('-inf')])
        self.position_min = np.array([float('inf'), float('inf'), float('inf')])

        self.dataset_type = dataset_type
        self.eval = eval

        # load three different datasets seperately
        if dataset_type == 'MeshRIR':
            self.load_mesh_rir(base_folder, eval, seq_len, fs)
        elif dataset_type == 'Treble':
            self.load_simu(base_folder, eval, seq_len, fs)
        else:
            raise ValueError("Unsupported dataset type")

        self.wave_chunks = torch.tensor(np.array(self.wave_chunks), dtype=torch.complex64)
        self.positions_rx = torch.tensor(np.array(self.positions_rx), dtype=torch.float32)
        self.positions_tx = torch.tensor(np.array(self.positions_tx), dtype=torch.float32)


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

