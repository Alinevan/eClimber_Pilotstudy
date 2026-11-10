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


def find_sync_peaks(t, x, lp_hz=None, invert=False):
    """Times of the SYNC_N_PEAKS most prominent peaks in the first SYNC_WINDOW_S seconds."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(t) & np.isfinite(x) & (t <= t[0] + cfg.SYNC_WINDOW_S)
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