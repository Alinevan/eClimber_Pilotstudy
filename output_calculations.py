"""Step 5: result tables and file output (add further outcome measures here)."""
import pandas as pd


def rms_table(data):
    """One row per direction x BWS x muscle with RMS and iEMG."""
    rows = []
    for direction, d in data.items():
        for bws, c in d["emg"].items():
            for m, r in c["rms"].items():
                rows.append({"direction": direction, "bws_percent": bws, "muscle": m, "rms": r,
                             "iemg": c["iemg"][m], "unit": c.get("unit") or "", "file": c["file"]})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["direction", "muscle", "bws_percent"])


def save_table(df, path):
    if df.empty:
        return
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def availability_table(data):
    """Which EMG / force files exist per direction and BWS level (missing levels are noted)."""
    import config as cfg
    rows = []
    for direction in cfg.DIRECTIONS:
        d = data.get(direction, {"emg": {}, "force": {}})
        levels = sorted(set(cfg.EXPECTED_BWS) | set(d["emg"]) | set(d["force"]))
        for bws in levels:
            e, f = d["emg"].get(bws), d["force"].get(bws)
            if e and f:
                note = "complete"
            elif not e and not f:
                note = "no data for this BWS level"
            elif not e:
                note = "EMG missing"
            else:
                note = "force missing"
            rows.append({"direction": direction, "bws_percent": bws,
                         "emg_file": e["file"] if e else "", "force_file": f["file"] if f else "",
                         "note": note})
    return pd.DataFrame(rows)