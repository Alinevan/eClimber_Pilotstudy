"""Step 2: EMG pre-processing (band-pass, rectify, envelope, RMS, iEMG)."""
from collections import defaultdict

import numpy as np
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt

import config as cfg
from utils import lowpass


def process_emg(sig, fs):
    """Band-pass, rectify, envelope. Returns (envelope, rms, iemg)."""
    sig = np.nan_to_num(sig - np.nanmean(sig))
    hi = min(cfg.BANDPASS_HZ[1], 0.45 * fs)  # stay below Nyquist
    b, a = butter(cfg.FILTER_ORDER, [cfg.BANDPASS_HZ[0] / (fs / 2.0), hi / (fs / 2.0)], btype="band")
    filtered = filtfilt(b, a, sig)
    envelope = lowpass(np.abs(filtered), cfg.ENVELOPE_LP_HZ, fs)
    rms = float(np.sqrt(np.mean(filtered ** 2)))
    iemg = float(simpson(envelope, dx=1.0 / fs))  # area under envelope, unit*s
    return envelope, rms, iemg


def group_muscles(channel_dict):
    """Apply MUSCLE_GROUP_MAP (average channels that share a group)."""
    if not cfg.MUSCLE_GROUP_MAP:
        return channel_dict
    groups = defaultdict(list)
    for name, sig in channel_dict.items():
        groups[cfg.MUSCLE_GROUP_MAP.get(name, name)].append(sig)
    return {g: np.mean(np.vstack(s), axis=0) for g, s in groups.items()}


def process_all(data):
    """Adds t, env, rms, iemg and duration to every EMG entry (in place)."""
    for direction, d in data.items():
        for bws, e in d["emg"].items():
            fs = e["fs"]
            chans = group_muscles(e["raw"])
            step = max(1, int(round(fs / cfg.PLOT_FS_HZ)))
            env, rms, iemg = {}, {}, {}
            for name, sig in chans.items():
                env_full, r, a = process_emg(sig, fs)
                env[name] = env_full[::step]   # decimated for plotting
                rms[name] = r
                iemg[name] = a
            e["t"] = np.arange(len(next(iter(env.values())))) * step / fs if env else np.array([])
            e["env"], e["rms"], e["iemg"] = env, rms, iemg
            e["duration"] = len(next(iter(chans.values()))) / fs if chans else 0.0
    return data