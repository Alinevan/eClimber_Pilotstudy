"""Step 3: shift each force trial onto the EMG time axis using 3 sync peaks."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import config as cfg
from utils import lowpass


def find_force_column(df, name):
    """Exact (case-insensitive) match first, then 'contains'."""
    target = name.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == target:
            return c
    return next((c for c in df.columns if target in str(c).strip().lower()), None)


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


def find_sync_peaks(t, x, lp_hz=None, invert=False):
    """Legacy peak detection using the fixed first ``SYNC_WINDOW_S`` seconds.

    This is intentionally retained for the original batch workflow. The manual
    synchronisation tool calls ``find_sync_peaks_in_window`` instead.
    """
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return None
    return _find_sync_peaks_in_window(t, x, t[0], t[0] + cfg.SYNC_WINDOW_S,
                                      lp_hz=lp_hz, invert=invert)


def find_sync_peaks_in_window(t, x, start_s, end_s, lp_hz=None, invert=False):
    """Find sync peaks in a user-selected time window."""
    if start_s is None or end_s is None:
        return None
    start_s, end_s = sorted((float(start_s), float(end_s)))
    if end_s <= start_s:
        return None
    return _find_sync_peaks_in_window(t, x, start_s, end_s, lp_hz=lp_hz, invert=invert)


def synchronize(data):
    """Shift each force trial onto the EMG time axis and crop it. Returns a report DataFrame."""
    report = []
    for direction, d in data.items():
        for bws, f in sorted(d["force"].items()):
            e = d["emg"].get(bws)
            row = {"direction": direction, "bws_percent": bws, "force_file": f["file"],
                   "emg_file": e["file"] if e else "", "synchronised": False}
            if e is None:
                print(f"  ! sync {direction} {bws}%: no EMG file, force left unsynchronised")
                report.append(row)
                continue

            te = find_sync_peaks(e["t"], e["env"][cfg.SYNC_EMG_LABEL]) \
                if cfg.SYNC_EMG_LABEL in e["env"] else None
            col = find_force_column(f["forces"], cfg.SYNC_FORCE_COLUMN)
            tf = find_sync_peaks(f["t"], f["forces"][col].to_numpy(), cfg.SYNC_FORCE_LP_HZ,
                                 cfg.SYNC_FORCE_INVERT) if col is not None else None

            offset = 0.0
            if te is not None and tf is not None:
                diffs = te - tf
                offset = float(np.mean(diffs))
                spread = float(np.ptp(diffs))
                row.update(synchronised=True, offset_s=offset, peak_offset_spread_s=spread,
                           emg_peaks_s=np.round(te, 3).tolist(), force_peaks_s=np.round(tf, 3).tolist())
                flag = "  ! CHECK: peaks disagree" if spread > cfg.SYNC_MAX_SPREAD_S else ""
                print(f"  sync {direction:10s} {bws:3d}%  force shifted {offset:+.3f} s "
                      f"(peak spread {spread * 1000:.0f} ms){flag}")
                e["sync_peaks"] = te
                f["sync_peaks"] = tf + offset
                f["sync_col"] = col
            else:
                why = []
                if cfg.SYNC_EMG_LABEL not in e["env"]:
                    why.append(f"EMG channel '{cfg.SYNC_EMG_LABEL}' not loaded")
                elif te is None:
                    why.append(f"<{cfg.SYNC_N_PEAKS} EMG peaks found")
                if col is None:
                    why.append(f"force column '{cfg.SYNC_FORCE_COLUMN}' not found")
                elif tf is None:
                    why.append(f"<{cfg.SYNC_N_PEAKS} force peaks found")
                print(f"  ! sync {direction} {bws}%: {'; '.join(why)}. Force left unsynchronised.")

            t_new = f["t"] + offset
            if cfg.CROP_TO_EMG:
                keep = np.flatnonzero((t_new >= 0) & (t_new <= e["duration"]))
                f["forces"] = f["forces"].iloc[keep].reset_index(drop=True)
                t_new = t_new[keep]
                row["cropped_to_s"] = e["duration"]
            f["t"] = t_new
            report.append(row)
    return pd.DataFrame(report)


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
                emg_keep = np.flatnonzero((e["t"] >= emg_start) & (e["t"] <= emg_end))
                e["t"] = e["t"][emg_keep]
                for channel in e.get("env", {}):
                    e["env"][channel] = e["env"][channel][emg_keep]
                row.update(emg_crop_start_s=emg_start, emg_crop_end_s=emg_end)

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
            row.update(synchronised=True, offset_s=offset)
            for key in ("peak_offset_spread_s", "emg_peaks_s", "force_peaks_s",
                        "emg_channel", "force_channel", "detection_mode"):
                if key in match:
                    row[key] = saved_row[key]
            rows.append(row)
    return pd.DataFrame(rows)
