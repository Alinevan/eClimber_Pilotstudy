"""Step 2: EMG pre-processing (band-pass, rectify, envelope, RMS, iEMG)."""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt

from . import config as cfg
from .utils import lowpass


def prepare_emg(sig, fs):
    """Band-pass and rectify one EMG channel.

    This is the preparation used before interactive cropping and
    synchronization. It deliberately does not calculate the final envelope
    or outcome measures.
    """
    sig = np.nan_to_num(sig - np.nanmean(sig))
    hi = min(cfg.BANDPASS_HZ[1], 0.45 * fs)  # stay below Nyquist
    b, a = butter(cfg.FILTER_ORDER, [cfg.BANDPASS_HZ[0] / (fs / 2.0), hi / (fs / 2.0)], btype="band")
    filtered = filtfilt(b, a, sig)
    return filtered, np.abs(filtered)


def calculate_window_metrics(filtered, fs, start_s=0.0, end_s=None, pad_s=1.0,
                             time_origin_s=0.0):
    """Calculate envelope, RMS, and iEMG for one time window.

    The optional padding is used only while calculating the low-pass envelope,
    reducing edge effects from ``filtfilt``. The returned envelope and all
    outcome measures cover only ``start_s`` through ``end_s``.
    """
    filtered = np.asarray(filtered, dtype=float)
    if filtered.size == 0:
        return np.array([]), float("nan"), float("nan")

    end_s = (time_origin_s + filtered.size / fs) if end_s is None else float(end_s)
    start_s = max(float(time_origin_s), float(start_s))
    end_s = min(time_origin_s + filtered.size / fs, end_s)
    if end_s <= start_s:
        return np.array([]), float("nan"), float("nan")

    start_idx = max(0, int(round((start_s - time_origin_s) * fs)))
    end_idx = min(filtered.size, int(round((end_s - time_origin_s) * fs)))
    if end_idx <= start_idx:
        return np.array([]), float("nan"), float("nan")

    pad = max(0, int(round(pad_s * fs)))
    padded_start = max(0, start_idx - pad)
    padded_end = min(filtered.size, end_idx + pad)
    envelope_padded = lowpass(np.abs(filtered[padded_start:padded_end]), cfg.ENVELOPE_LP_HZ, fs)
    envelope = envelope_padded[start_idx - padded_start:end_idx - padded_start]
    selected = filtered[start_idx:end_idx]
    rms = float(np.sqrt(np.mean(selected ** 2)))
    iemg = float(simpson(envelope, dx=1.0 / fs))  # area under envelope, unit*s
    return envelope, rms, iemg


def channel_side_and_muscle(channel):
    """Return a channel's side (``L``/``R``) and side-independent muscle key."""
    channel = str(channel)
    if len(channel) >= 2 and channel[0] in {"L", "R"} and channel[1] == ".":
        return channel[0], channel[2:]
    return "", channel


def moving_average_peak(values, fs, window_s=None):
    """Return the maximum moving-average value used for EMG normalization."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    window_s = (cfg.NORMALIZATION_MOVING_AVERAGE_S
                if window_s is None else float(window_s))
    window_samples = max(1, int(round(window_s * float(fs))))
    if values.size < window_samples:
        return float(np.mean(values))
    kernel = np.ones(window_samples, dtype=float) / window_samples
    moving = np.convolve(values, kernel, mode="valid")
    return float(np.max(moving))


def calculate_cci(biceps_envelope, triceps_envelope, fs):
    """Calculate the paper's biceps-triceps co-contraction index."""
    biceps_envelope = np.asarray(biceps_envelope, dtype=float)
    triceps_envelope = np.asarray(triceps_envelope, dtype=float)
    n = min(len(biceps_envelope), len(triceps_envelope))
    if n == 0:
        return float("nan")
    biceps_envelope = biceps_envelope[:n]
    triceps_envelope = triceps_envelope[:n]
    finite = np.isfinite(biceps_envelope) & np.isfinite(triceps_envelope)
    if not finite.any():
        return float("nan")
    biceps_envelope = biceps_envelope[finite]
    triceps_envelope = triceps_envelope[finite]
    denominator = float(simpson(biceps_envelope + triceps_envelope, dx=1.0 / fs))
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    overlap = float(simpson(np.minimum(biceps_envelope, triceps_envelope), dx=1.0 / fs))
    return float(100.0 * 2.0 * overlap / denominator)


def calculate_normalized_outcomes(data):
    """Add paper-style normalized EMG outcomes to all participant trials.

    A common reference is calculated for each side-independent muscle across
    all available directions, BWS levels, and left/right channels of the
    participant. The envelope reference is the maximum 0.5 s moving average;
    the RMS reference is the largest raw RMS value.
    """
    envelope_peaks = {}
    rms_references = {}

    for _direction, d in data.items():
        for _bws, entry in d.get("emg", {}).items():
            for channel, envelope in entry.get("env_window", {}).items():
                _side, muscle = channel_side_and_muscle(channel)
                peak = moving_average_peak(envelope, entry["fs"])
                if np.isfinite(peak):
                    envelope_peaks[muscle] = max(envelope_peaks.get(muscle, -np.inf), peak)
                rms = entry.get("rms", {}).get(channel, np.nan)
                if np.isfinite(rms):
                    rms_references[muscle] = max(rms_references.get(muscle, -np.inf), float(rms))

    for _direction, d in data.items():
        for _bws, entry in d.get("emg", {}).items():
            entry["env_normalized_window"] = {}
            entry["mean_envelope_normalized"] = {}
            entry["rms_normalized"] = {}
            entry["iemg_normalized"] = {}
            entry["normalization_peaks"] = {}
            entry["rms_references"] = {}
            for channel, envelope in entry.get("env_window", {}).items():
                _side, muscle = channel_side_and_muscle(channel)
                peak = envelope_peaks.get(muscle, np.nan)
                rms_reference = rms_references.get(muscle, np.nan)
                if np.isfinite(peak) and peak > 0:
                    normalized = 100.0 * np.asarray(envelope, dtype=float) / peak
                    entry["env_normalized_window"][channel] = normalized
                    entry["mean_envelope_normalized"][channel] = float(np.mean(normalized))
                    entry["iemg_normalized"][channel] = float(
                        simpson(normalized, dx=1.0 / float(entry["fs"]))
                    )
                else:
                    entry["env_normalized_window"][channel] = np.full_like(
                        np.asarray(envelope, dtype=float), np.nan
                    )
                    entry["mean_envelope_normalized"][channel] = float("nan")
                    entry["iemg_normalized"][channel] = float("nan")
                entry["normalization_peaks"][channel] = peak
                if np.isfinite(rms_reference) and rms_reference > 0:
                    entry["rms_normalized"][channel] = float(
                        100.0 * entry["rms"].get(channel, np.nan) / rms_reference
                    )
                else:
                    entry["rms_normalized"][channel] = float("nan")
                entry["rms_references"][channel] = rms_reference

    return envelope_peaks, rms_references


def process_emg(sig, fs):
    """Process one complete channel when explicitly requested by a caller."""
    filtered, _rectified = prepare_emg(sig, fs)
    envelope, rms, iemg = calculate_window_metrics(filtered, fs)
    return envelope, rms, iemg


def group_muscles(channel_dict):
    """Apply MUSCLE_GROUP_MAP (average channels that share a group)."""
    if not cfg.MUSCLE_GROUP_MAP:
        return channel_dict
    groups = defaultdict(list)
    for name, sig in channel_dict.items():
        groups[cfg.MUSCLE_GROUP_MAP.get(name, name)].append(sig)
    return {g: np.mean(np.vstack(s), axis=0) for g, s in groups.items()}


def process_all(data, calculate_outcomes=True):
    """Prepare all EMG entries, optionally calculating full-record outcomes.

    ``calculate_outcomes=False`` is used by the interactive synchronization
    workflow: it stores full-recording filtered and rectified signals for
    viewing and peak-window selection, but does not calculate envelopes, RMS,
    or iEMG before the climbing window has been chosen.
    """
    for direction, d in data.items():
        for bws, e in d["emg"].items():
            fs = e["fs"]
            chans = group_muscles(e["raw"])
            step = max(1, int(round(fs / cfg.PLOT_FS_HZ)))
            filtered, rectified = {}, {}
            env, rms, iemg = {}, {}, {}
            for name, sig in chans.items():
                filtered_full, rectified_full = prepare_emg(sig, fs)
                filtered[name] = filtered_full
                rectified[name] = rectified_full
                if calculate_outcomes:
                    env_full, r, a = calculate_window_metrics(filtered_full, fs)
                    env[name] = env_full[::step]   # decimated for plotting
                    rms[name] = r
                    iemg[name] = a
            e["filtered"], e["rectified"] = filtered, rectified
            if calculate_outcomes:
                e["t"] = np.arange(len(next(iter(env.values())))) * step / fs if env else np.array([])
                e["env"], e["rms"], e["iemg"] = env, rms, iemg
            else:
                e["t"] = np.arange(len(next(iter(rectified.values())))) / fs if rectified else np.array([])
                e.pop("env", None)
                e.pop("rms", None)
                e.pop("iemg", None)
            e["duration"] = len(next(iter(chans.values()))) / fs if chans else 0.0
    return data


def calculate_outcomes_from_sync_report(data, report):
    """Calculate EMG outcomes only inside saved analysis windows.

    The report must contain ``direction``, ``bws_percent``,
    ``analysis_start_s``, and ``analysis_end_s``. Conditions without a valid
    window receive empty outcome dictionaries and are returned in ``issues``;
    they are never silently calculated over the full recording.
    """
    issues = []
    if report is None or report.empty:
        report = None

    for direction, d in data.items():
        for bws, entry in d.get("emg", {}).items():
            entry["env"], entry["rms"], entry["iemg"] = {}, {}, {}
            entry["env_window"], entry["mean_envelope"] = {}, {}
            entry["window_duration_s"] = float("nan")
            if report is None:
                issues.append(f"{direction} {bws}%: no synchronization report")
                continue

            report_bws = pd.to_numeric(report["bws_percent"], errors="coerce")
            matches = report[(report["direction"].astype(str) == str(direction)) &
                             (report_bws == bws)]
            if matches.empty:
                issues.append(f"{direction} {bws}%: no report row")
                continue
            row = matches.iloc[-1]
            try:
                start_s = float(row["analysis_start_s"])
                end_s = float(row["analysis_end_s"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{direction} {bws}%: analysis window missing")
                continue
            if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
                issues.append(f"{direction} {bws}%: invalid analysis window")
                continue

            fs = float(entry["fs"])
            filtered = entry.get("filtered", {})
            if not filtered:
                issues.append(f"{direction} {bws}%: filtered EMG unavailable")
                continue

            step = max(1, int(round(fs / cfg.PLOT_FS_HZ)))
            for channel, signal in filtered.items():
                time_origin = float(entry.get("time_origin_s", 0.0))
                envelope, rms, iemg = calculate_window_metrics(
                    signal, fs, start_s, end_s, time_origin_s=time_origin
                )
                start_idx = max(0, int(round((start_s - time_origin) * fs)))
                end_idx = min(len(signal), int(round((end_s - time_origin) * fs)))
                env_full = np.full(len(signal), np.nan, dtype=float)
                env_full[start_idx:end_idx] = envelope
                entry["env"][channel] = env_full[::step]
                entry["env_window"][channel] = envelope
                entry["mean_envelope"][channel] = float(np.mean(envelope)) if len(envelope) else float("nan")
                entry["rms"][channel] = rms
                entry["iemg"][channel] = iemg

            time_origin = float(entry.get("time_origin_s", 0.0))
            entry["t"] = time_origin + np.arange(len(next(iter(entry["env"].values())))) * step / fs
            entry["analysis_start_s"] = start_s
            entry["analysis_end_s"] = end_s
            entry["window_duration_s"] = float(sum(len(v) for v in entry["env_window"].values()) /
                                                 max(1, len(entry["env_window"])) / fs)

    calculate_normalized_outcomes(data)

    return issues
