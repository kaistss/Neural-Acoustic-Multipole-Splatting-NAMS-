"""
Evaluation metrics in this module are adapted from the Acoustic Volume Rendering (AVR) codebase.

- Repository: https://github.com/penn-waves-lab/AVR
- Reference: Lan, Zitong, et al. "Acoustic volume rendering for neural impulse response fields."
  Advances in Neural Information Processing Systems 37 (2024): 44600-44623.
"""
import numpy as np
from scipy import stats
from scipy.signal import hilbert
import scipy
import auraloss
import torch

def metric_cal(ori_ir, pred_ir, fs=48000, window=32):
    """calculate the evaluation metric

    Parameters
    ----------
    ori_ir : np.array
        ground truth impulse response
    pred_ir : np.array
        predicted impulse response
    fs : int
        sampling rate, by default 48000

    Returns
    -------
    evaluation metrics
    """
    
    if ori_ir.ndim == 1:
        ori_ir = ori_ir[np.newaxis, :]
    if pred_ir.ndim == 1:
        pred_ir = pred_ir[np.newaxis, :]

    # prevent numerical issue for log calculation
    multi_stft = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1, fft_sizes=[512, 256, 128], win_lengths=[300, 150, 75], hop_sizes=[60, 30, 8])
    multi_stft_loss = multi_stft(torch.tensor(ori_ir).unsqueeze(1), torch.tensor(pred_ir).unsqueeze(1))

    fft_ori = np.fft.fft(ori_ir, axis=-1)
    fft_predict = np.fft.fft(pred_ir, axis=-1)

    angle_error = np.mean(np.abs(np.cos(np.angle(fft_ori)) - np.cos(np.angle(fft_predict)))) + np.mean(np.abs(np.sin(np.angle(fft_ori)) - np.sin(np.angle(fft_predict))))
    amp_ori = scipy.ndimage.convolve1d(np.abs(fft_ori), np.ones(window))
    amp_predict = scipy.ndimage.convolve1d(np.abs(fft_predict), np.ones(window))
    amp_error = np.mean(np.abs(amp_ori - amp_predict) / amp_ori)

    # calculate the envelop error
    ori_env = np.abs(hilbert(ori_ir))
    pred_env = np.abs(hilbert(pred_ir))
    env_error = np.mean(np.abs(ori_env - pred_env) / np.max(ori_env, axis=1, keepdims=True))

    # derevie the energy trend
    ori_energy = 10.0 * np.log10(np.cumsum(ori_ir[:,::-1]**2 + 1e-9, axis=-1)[:,::-1])
    pred_energy = 10.0 * np.log10(np.cumsum(pred_ir[:,::-1]**2 + 1e-9, axis=-1)[:,::-1])

    ori_energy -= ori_energy[:, 0].reshape(-1, 1)
    pred_energy -= pred_energy[:, 0].reshape(-1, 1)
    
    # calculate the t60 percentage error and EDT time error
    ori_t60, ori_edt = t60_EDT_cal(ori_energy, fs=fs)
    pred_t60, pred_edt = t60_EDT_cal(pred_energy, fs=fs)
    t60_error = np.mean(np.abs(ori_t60 - pred_t60) / ori_t60)
    edt_error = np.mean(np.abs(ori_edt - pred_edt))

    # calculate the C50 error
    base_sample = 0
    samples_50ms = int(0.05 * fs) + base_sample  # Number of samples in 50 ms
    # Compute the energy in the first 50ms and from 50ms to the end
    energy_ori_early = np.sum(ori_ir[:,base_sample:samples_50ms]**2, axis=-1)
    energy_ori_late = np.sum(ori_ir[:,samples_50ms:]**2, axis=-1)
    energy_pred_early = np.sum(pred_ir[:,base_sample:samples_50ms]**2, axis=-1)
    energy_pred_late = np.sum(pred_ir[:,samples_50ms:]**2, axis=-1)

    # Calculate C50 for the original and predicted impulse response
    C50_ori = 10.0 * np.log10(energy_ori_early / energy_ori_late)
    C50_pred = 10.0 * np.log10(energy_pred_early / energy_pred_late)
    C50_error = np.mean(np.abs(C50_ori - C50_pred))

    return angle_error, amp_error, env_error, t60_error, edt_error, C50_error, multi_stft_loss, ori_energy, pred_energy


def t60_EDT_cal(energys, init_db=-5, end_db=-25, factor=3.0, fs=48000):
    """calculate the T60 and EDT metric of the given impulse response normalized energy trend
    t60: find the time it takes to decay from -5db to -65db.
        A usual way to do this is to calculate the time it takes from -5 to -25db, and multiply by 3.0
    
    EDT: Early decay time, time it takes to decay from 0db to -10db, and multiply the number by 6

    Parameters
    ----------
    energys : np.array
        normalized energy
    init_db : int, optional
        t60 start db, by default -5
    end_db : int, optional
        t60 end db, by default -25
    factor : float, optional
        t60 multiply factor, by default 3.0
    fs : int, optional
        sampling rate, by default 48000

    Returns
    -------
    t60 : float
    edt : float, seconds
    """

    t60_all = []
    edt_all = []

    for energy in energys:
        # find the -10db point
        edt_factor = 6.0
        energy_n10db = energy[np.abs(energy - (-10)).argmin()]

        n10db_sample = np.where(energy == energy_n10db)[0][0]
        edt = n10db_sample / fs * edt_factor

        # find the intersection of -5db and -25db position
        energy_init = energy[np.abs(energy - init_db).argmin()]
        energy_end = energy[np.abs(energy - end_db).argmin()]
        init_sample = np.where(energy == energy_init)[0][0]
        end_sample = np.where(energy == energy_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = energy[init_sample:end_sample + 1]
        
        # regress to find the db decay trend
        slope, intercept = stats.linregress(x, y)[0:2]
        db_regress_init = (init_db - intercept) / slope
        db_regress_end = (end_db - intercept) / slope

        # get t60 value
        t60 = factor * (db_regress_end - db_regress_init)
        
        t60_all.append(t60)
        edt_all.append(edt)

    t60_all = np.array(t60_all)
    edt_all = np.array(edt_all)

    return t60_all, edt_all

