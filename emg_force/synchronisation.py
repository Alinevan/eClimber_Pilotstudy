"""Manual-report-based EMG/force synchronisation.

The active workflow detects peaks and position dips only inside windows chosen
in :mod:`manual_synchronisation`, saves those choices, and applies them with
``synchronize_from_report``. The former fixed-window automatic synchronizer
is intentionally not part of this module anymore.
"""
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from . import config as cfg
from .utils import lowpass


def _find_sync_peaks_in_window(t, x, start_s, end_s, lp_hz=None, invert=False):
    """Times of the most prominent peaks inside an explicitly selected window."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(t) & np.isfinite(x) & (t >= start_s) & (t <= end_s)
    t, x = t[m], x[m]
    if len(t) < 10:
        return None
    fs = 1.0 / np.median(np.diff(t))
    if lp_hz and fs > 2.5 * lp_hz:
        x = lowpass(x, lp_hz, fs)
    if invert:
        x = -x
    rng = np.ptp(x)
    if rng == 0:
        return None
    idx, props = find_peaks(x, distance=max(1, int(cfg.SYNC_MIN_PEAK_DISTANCE_S * fs)),
                            prominence=0.1 * rng)
    if len(idx) < cfg.SYNC_N_PEAKS:
        return None
    top = idx[np.argsort(props["prominences"])[-cfg.SYNC_N_PEAKS:]]
    return np.sort(t[top])


def find_sync_peaks_in_window(t, x, start_s, end_s, lp_hz=None, invert=False):
    """Find sync peaks in a user-selected time window."""
    if start_s is None or end_s is None:
        return None
    start_s, end_s = sorted((float(start_s), float(end_s)))
    if end_s <= start_s:
        return None
    return _find_sync_peaks_in_window(t, x, start_s, end_s, lp_hz=lp_hz, invert=invert)


def emg_sync_signal(t, x, lp_hz=None):
    """Smooth only the signal supplied by the caller for synchronization.

    Callers pass the manually selected detection window, so this function does
    not filter the unused portions of the recording.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(t) & np.isfinite(x)
    t, x = t[valid], x[valid]
    if len(t) < 10:
        return t, x
    order = np.argsort(t)
    t, x = t[order], x[order]
    x = np.nan_to_num(x - np.nanmin(x))
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if lp_hz and len(dt) and 1.0 / np.median(dt) > 2.5 * lp_hz:
        x = lowpass(x, lp_hz, 1.0 / np.median(dt))
    return t, x


def find_emg_sync_peaks_in_window(t, x, start_s, end_s, lp_hz=None):
    """Find exactly three robust EMG sync peaks inside a selected window.

    EMG synchronization contractions can be close together and may have
    different amplitudes. The EMG path therefore uses adaptive prominence and
    distance thresholds, then falls back to the three strongest local maxima
    if the initial threshold is too strict. Force/position detection keeps its
    separate, stricter detector above.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(t) & np.isfinite(x) & (t >= start_s) & (t <= end_s)
    t, x = t[valid], x[valid]
    if len(t) < 10:
        return None
    t, x = emg_sync_signal(t, x, lp_hz=lp_hz)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return None
    fs = 1.0 / np.median(dt)
    amplitude = float(np.ptp(x))
    if not np.isfinite(amplitude) or amplitude <= 0:
        return None

    # Start with a conservative detector, then relax only if fewer than three
    # events are found. The selected window is the user's primary constraint.
    distances_s = (
        cfg.SYNC_EMG_MIN_PEAK_DISTANCE_S,
        0.25,
    )
    prominence_fractions = (0.10, 0.05, 0.02, 0.01)
    for distance_s in distances_s:
        distance = max(1, int(round(distance_s * fs)))
        for fraction in prominence_fractions:
            idx, props = find_peaks(
                x,
                distance=distance,
                prominence=max(fraction * amplitude, np.finfo(float).eps),
            )
            if len(idx) >= cfg.SYNC_N_PEAKS:
                chosen = idx[np.argsort(props["prominences"])[-cfg.SYNC_N_PEAKS:]]
                return np.sort(t[chosen])

    # Final fallback: no prominence cut, ranking local maxima by height. This
    # still returns only peaks inside the manually selected EMG window.
    idx, _ = find_peaks(x, distance=max(1, int(round(0.20 * fs))))
    if len(idx) >= cfg.SYNC_N_PEAKS:
        chosen = idx[np.argsort(x[idx])[-cfg.SYNC_N_PEAKS:]]
        return np.sort(t[chosen])

    # If smoothing produced broad plateaus, select one maximum from each of
    # three consecutive parts of the selected window. This guarantees a clear
    # three-event result for a window that the user deliberately supplied.
    parts = np.array_split(np.arange(len(x)), cfg.SYNC_N_PEAKS)
    selected = [part[np.argmax(x[part])] for part in parts if len(part)]
    if len(selected) == cfg.SYNC_N_PEAKS:
        return np.sort(t[np.asarray(selected, dtype=int)])
    return None


def suggest_analysis_window(direction, t, position, position_dips_s, offset_s=0.0):
    """Suggest the common EMG/force calculation window.

    The window starts one second after the synchronized final position dip.
    Vertical trials end at their post-synchronization global position maximum.
    Horizontal trials end when position returns close to the baseline around
    the final dip and stays there. A fallback to the recording end is returned
    for horizontal trials when no sustained return is detected, so the user
    can correct only the endpoint in the manual interface.
    """
    if position_dips_s is None or len(position_dips_s) == 0:
        return None

    t = np.asarray(t, dtype=float)
    position = np.asarray(position, dtype=float)
    valid = np.isfinite(t) & np.isfinite(position)
    if valid.sum() < 10:
        return None
    t, position = t[valid], position[valid]
    order = np.argsort(t)
    t, position = t[order], position[order]
    fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 0.0
    smoothed = position
    if cfg.SYNC_FORCE_LP_HZ and fs > 2.5 * cfg.SYNC_FORCE_LP_HZ:
        smoothed = lowpass(position, cfg.SYNC_FORCE_LP_HZ, fs)

    last_dip_raw = float(np.asarray(position_dips_s, dtype=float)[-1])
    last_dip_aligned = last_dip_raw + float(offset_s)
    start_s = last_dip_aligned + float(cfg.CALCULATION_START_DELAY_S)
    start_idx = int(np.searchsorted(t + offset_s, start_s, side="left"))
    if start_idx >= len(t) - 2:
        return None

    # Estimate the starting level around the final synchronization valley,
    # rather than at ``start_idx`` (which is already one second into the
    # climbing movement). This is the reference needed for horizontal trials.
    dip_idx = int(np.argmin(np.abs((t + offset_s) - last_dip_aligned)))
    baseline_half_window = max(1, int(round(0.20 * fs)))
    baseline_values = smoothed[max(0, dip_idx - baseline_half_window):
                               min(len(smoothed), dip_idx + baseline_half_window + 1)]
    baseline = float(np.median(baseline_values))

    post = smoothed[start_idx:]
    plateau = float(np.percentile(post, 90))
    excursion = plateau - baseline
    if not np.isfinite(excursion) or excursion <= 0:
        return None

    if str(direction).lower() != "horizontal":
        # Vertical trials retain the successful global-maximum rule.
        peak_idx = start_idx + int(np.argmax(post))
        end_s = float(t[peak_idx] + offset_s)
        if end_s <= start_s:
            return None
        return {
            "analysis_start_s": start_s,
            "analysis_end_s": end_s,
            "analysis_window_method": (
                "vertical: one second after last position dip to global position maximum"
            ),
            "analysis_position_baseline": baseline,
            "analysis_position_tolerance": max(0.05 * excursion, 1e-12),
            "analysis_return_detected": True,
        }

    # For horizontal trials, wait until the signal has reached the elevated
    # plateau before searching for the final downward return. This avoids
    # treating short fluctuations immediately after the sync event as the end.
    high_threshold = baseline + 0.50 * excursion
    return_threshold = baseline + min(0.30 * excursion, 0.45 * excursion)
    reached_high = np.flatnonzero(post >= high_threshold)
    min_return_samples = max(1, int(round(0.40 * fs)))
    end_idx = None
    if len(reached_high):
        search_start = int(reached_high[0])
        below = post[search_start:] <= return_threshold
        for candidate in range(0, max(0, len(below) - min_return_samples + 1)):
            if np.all(below[candidate:candidate + min_return_samples]):
                end_idx = start_idx + search_start + candidate
                break

    if end_idx is None:
        # Keep the automatically calculated start, but expose the endpoint as
        # an explicit manual correction in the interactive position plot.
        return {
            "analysis_start_s": start_s,
            "analysis_end_s": float(t[-1] + offset_s),
            "analysis_window_method": (
                "horizontal: return not detected; endpoint requires manual review"
            ),
            "analysis_position_baseline": baseline,
            "analysis_position_tolerance": return_threshold - baseline,
            "analysis_return_detected": False,
        }

    end_s = float(t[end_idx] + offset_s)
    if end_s <= start_s:
        return None
    return {
        "analysis_start_s": start_s,
        "analysis_end_s": end_s,
        "analysis_window_method": (
            "horizontal: one second after last position dip to sustained return toward baseline"
        ),
        "analysis_position_baseline": baseline,
        "analysis_position_tolerance": return_threshold - baseline,
        "analysis_return_detected": True,
    }


def synchronize_from_report(data, report_path):
    """Apply offsets selected in ``manual_synchronisation.py``.

    Conditions not present in the manual report are left unsynchronised and
    explicitly marked in the returned report. This avoids silently falling
    back to the fixed-window detector for a condition the user intended to
    review manually.
    """
    saved = pd.read_csv(report_path)
    required = {"direction", "bws_percent", "offset_s"}
    missing = required - set(saved.columns)
    if missing:
        raise ValueError(f"Manual sync report is missing columns: {sorted(missing)}")

    rows = []
    for direction, d in data.items():
        for bws, f in sorted(d["force"].items()):
            e = d["emg"].get(bws)
            row = {"direction": direction, "bws_percent": bws, "force_file": f["file"],
                   "emg_file": e["file"] if e else "", "synchronised": False,
                   "sync_method": "manual_report"}
            match = saved[(saved["direction"].astype(str) == str(direction)) &
                          (pd.to_numeric(saved["bws_percent"], errors="coerce") == bws)]
            if e is None:
                row["reason"] = "EMG missing"
                rows.append(row)
                continue
            if match.empty:
                row["reason"] = "manual offset missing"
                rows.append(row)
                continue

            saved_row = match.iloc[-1]

            def saved_float(name, default=None):
                if name not in saved_row or pd.isna(saved_row[name]):
                    return default
                return float(saved_row[name])

            emg_start = saved_float("emg_crop_start_s")
            emg_end = saved_float("emg_crop_end_s")
            if emg_start is not None and emg_end is not None:
                # Keep the full preprocessed EMG time base in memory. The
                # saved crop is metadata for display/reuse; outcome metrics
                # are calculated later from the saved analysis window.
                row.update(emg_crop_start_s=emg_start, emg_crop_end_s=emg_end)
                # Keep the crop bounds available to the final plotting stage.
                # The plot uses them to rebase every EMG condition to its own
                # crop start, without changing the calculation time base.
                e["crop_start_s"] = emg_start
                e["crop_end_s"] = emg_end
            else:
                emg_time = np.asarray(e.get("t", []), dtype=float)
                if len(emg_time):
                    e["crop_start_s"] = float(emg_time[0])
                    e["crop_end_s"] = float(emg_time[-1])

            force_start = saved_float("force_crop_start_s")
            force_end = saved_float("force_crop_end_s")
            if force_start is not None and force_end is not None:
                force_keep = np.flatnonzero((f["t"] >= force_start) & (f["t"] <= force_end))
                f["t"] = f["t"][force_keep]
                f["forces"] = f["forces"].iloc[force_keep].reset_index(drop=True)
                row.update(force_crop_start_s=force_start, force_crop_end_s=force_end)

            offset = float(saved_row["offset_s"])
            t_new = f["t"] + offset
            if cfg.CROP_TO_EMG:
                crop_start = emg_start if emg_start is not None else 0.0
                crop_end = emg_end if emg_end is not None else e["duration"]
                keep = np.flatnonzero((t_new >= crop_start) & (t_new <= crop_end))
                f["forces"] = f["forces"].iloc[keep].reset_index(drop=True)
                t_new = t_new[keep]
                row["cropped_to_s"] = crop_end
            f["t"] = t_new
            if force_start is not None and force_end is not None:
                f["crop_start_s"] = force_start + offset
                f["crop_end_s"] = force_end + offset
            row.update(synchronised=True, offset_s=offset)

            def saved_list(name):
                if name not in saved_row or pd.isna(saved_row[name]):
                    return None
                value = saved_row[name]
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        return None
                return np.asarray(value, dtype=float)

            for key in ("peak_offset_spread_s", "emg_peaks_s", "force_peaks_s",
                        "position_dips_s", "emg_channel", "force_channel",
                        "position_channel", "detection_mode", "analysis_start_s",
                        "analysis_end_s", "analysis_window_method",
                        "analysis_position_baseline", "analysis_position_tolerance",
                        "analysis_return_detected"):
                if key in match and not pd.isna(saved_row[key]):
                    row[key] = saved_row[key]

            emg_peaks = saved_list("emg_peaks_s")
            force_peaks = saved_list("force_peaks_s")
            if emg_peaks is not None:
                e["sync_peaks"] = emg_peaks
                e["sync_channel"] = saved_row.get("emg_channel", cfg.SYNC_EMG_LABEL)
            if force_peaks is not None:
                f["sync_peaks"] = force_peaks + offset
                f["sync_col"] = saved_row.get("force_channel", cfg.SYNC_FORCE_COLUMN)
            position_dips = saved_list("position_dips_s")
            if position_dips is not None:
                f["position_sync_peaks"] = position_dips + offset
                f["position_col"] = saved_row.get("position_channel", saved_row.get("force_channel"))
            rows.append(row)
    return pd.DataFrame(rows)
