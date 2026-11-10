"""Step 1: read raw EMG (.c3d) and force (.csv) files. No signal processing here."""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ezc3d
except ImportError:
    raise SystemExit("ezc3d is required: pip install ezc3d")

import config as cfg
from utils import clean_label, lowpass

_KEYWORDS = "|".join(sorted(map(re.escape, cfg.DIRECTION_ALIASES), key=len, reverse=True))  # longest first
_FNAME_RE = re.compile(rf"(?:^|[_\-\s])({_KEYWORDS})[_\-\s]*(?:bws)?[_\-\s]*(\d+)", re.IGNORECASE)


_PARTICIPANT_RE = re.compile(rf"_{re.escape(cfg.PARTICIPANT_PREFIX)}0*(\d+)$", re.IGNORECASE)


def parse_participant(path: Path):
    """Return the participant number from the end of the file name (e.g. ..._P03 -> 3), or None."""
    m = _PARTICIPANT_RE.search(path.stem)
    return int(m.group(1)) if m else None


def find_subfolder(root: Path, keyword: str):
    """First subfolder of root whose name contains keyword (case-insensitive), or None."""
    for d in sorted(root.iterdir()):
        if d.is_dir() and keyword.lower() in d.name.lower():
            return d
    return None


def find_data_folders(root: Path):
    """Returns (emg_dir, force_dir); stops with a clear message if one is missing."""
    emg_dir = find_subfolder(root, cfg.EMG_FOLDER_KEYWORD)
    force_dir = find_subfolder(root, cfg.FORCE_FOLDER_KEYWORD)
    for name, d, kw in (("EMG", emg_dir, cfg.EMG_FOLDER_KEYWORD), ("force", force_dir, cfg.FORCE_FOLDER_KEYWORD)):
        if d is None:
            raise SystemExit(f"No {name} folder (name containing '{kw}') found in {root}")
    return emg_dir, force_dir


def list_participants(emg_dir: Path, force_dir: Path):
    """Sorted participant numbers found in the EMG and force folders."""
    ids = set()
    for folder, pattern in ((emg_dir, "*.c3d"), (force_dir, "*.csv")):
        for f in folder.glob(pattern):
            pid = parse_participant(f)
            if pid is not None:
                ids.add(pid)
    return sorted(ids)


def parse_condition(path: Path):
    """Return (direction, bws_percent) from the file name (e.g. Lat50_P01 -> horizontal, 50), or None."""
    m = _FNAME_RE.search(path.stem)
    if not m:
        return None
    return cfg.DIRECTION_ALIASES[m.group(1).lower()], int(m.group(2))


def load_emg(path: Path):
    """Read EMG channels from a C3D. Returns (fs, {label: raw signal}, unit)."""
    c = ezc3d.c3d(str(path))
    labels = c["parameters"]["ANALOG"]["LABELS"]["value"]
    fs = float(c["header"]["analogs"]["frame_rate"])
    data = c["data"]["analogs"][0]  # (n_channels, n_samples)
    try:
        units = c["parameters"]["ANALOG"]["UNITS"]["value"]
    except KeyError:
        units = []

    wanted = {clean_label(l) for l in cfg.EMG_LABELS}
    channels, unit = {}, None
    for i, (raw_label, sig) in enumerate(zip(labels, data)):
        name = clean_label(raw_label)
        if wanted:
            if name not in wanted:
                continue
        elif any(k.lower() in name.lower() for k in cfg.EMG_EXCLUDE_KEYWORDS):
            continue
        if unit is None and i < len(units) and units[i].strip():
            unit = units[i].strip()
        base, k = name, 2
        while name in channels:  # keep duplicate labels distinct
            name = f"{base}_{k}"
            k += 1
        channels[name] = np.asarray(sig, dtype=float)

    if wanted and (missing := wanted - set(channels)):
        print(f"  ! {path.name}: labels not found: {sorted(missing)}. Available: {labels}")
    return fs, channels, unit


def load_force(path: Path):
    """Read a force CSV. Returns (time_s, DataFrame of force columns)."""
    df = pd.read_csv(path, sep=None, engine="python")  # auto-detect , ; or tab
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="all")

    time_col = next((c for c in df.columns
                     if str(c).strip().lower() in ("t", "time", "time_s", "time (s)", "times", "timestamp")
                     or str(c).strip().lower().startswith("time")), None)
    if time_col is not None:
        t = df[time_col].to_numpy(dtype=float)
        t = t - t[0]
        if np.nanmax(t) > 1e4:  # looks like milliseconds
            t = t / 1000.0
        forces = df.drop(columns=[time_col])
    else:
        t = np.arange(len(df)) / cfg.FORCE_FS_DEFAULT
        forces = df

    forces = forces.loc[:, [c for c in forces.columns if not str(c).lower().startswith("unnamed")]]
    if cfg.FORCE_LP_HZ:
        fs = 1.0 / np.nanmedian(np.diff(t))
        forces = forces.apply(lambda col: lowpass(col.interpolate().to_numpy(), cfg.FORCE_LP_HZ, fs))
    return t, forces


def load_all(emg_dir: Path, force_dir: Path, participant: int):
    """Raw data of ONE participant: {direction: {"emg": {bws: {...}}, "force": {bws: {...}}}}."""
    out = defaultdict(lambda: {"emg": {}, "force": {}})

    for f in sorted(emg_dir.glob("*.c3d")):
        if parse_participant(f) != participant:
            continue
        cond = parse_condition(f)
        if cond is None:
            print(f"  skip (name not recognised): {f.name}")
            continue
        direction, bws = cond
        fs, chans, unit = load_emg(f)
        out[direction]["emg"][bws] = {"fs": fs, "raw": chans, "unit": unit, "file": f.name}
        print(f"  EMG   {direction:10s} {bws:3d}%  {len(chans)} channels @ {fs:.0f} Hz  ({f.name})")

    for f in sorted(force_dir.glob("*.csv")):
        if parse_participant(f) != participant:
            continue
        cond = parse_condition(f)
        if cond is None:
            print(f"  skip (name not recognised): {f.name}")
            continue
        direction, bws = cond
        t, forces = load_force(f)
        out[direction]["force"][bws] = {"t": t, "forces": forces, "file": f.name}

    return out