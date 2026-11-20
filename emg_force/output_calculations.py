"""Result tables and file output for EMG and force outcomes."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common.participants import participant_tag

from . import config as cfg
from . import emg_processing


def _participant_number(value):
    """Parse an Excel participant cell such as ``P01`` or ``1``."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"[Pp]?0*(\d+)", text)
    return int(match.group(1)) if match else None


def _bws_from_header(value):
    """Parse Excel headers written as 0.25, 25, or ``25%``."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    else:
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        numeric = float(match.group(0))
    if 0 < numeric <= 1:
        numeric *= 100.0
    numeric = int(round(numeric))
    return numeric if numeric in cfg.EXPECTED_BWS else None


def load_nominal_force_values(force_dir):
    """Read target settings in kg from the participant/BWS Excel lookup table."""
    path = Path(force_dir) / cfg.NOMINAL_FORCE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Nominal-force lookup file not found: {path}")

    raw = pd.read_excel(path, header=None)
    if raw.empty or raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError(f"Nominal-force lookup file is empty or incomplete: {path}")

    columns = {}
    for column in range(1, raw.shape[1]):
        bws = _bws_from_header(raw.iat[0, column])
        if bws is not None:
            columns[bws] = column
    if not columns:
        raise ValueError(f"No 25%, 50%, or 75% headers found in {path}")

    values = {}
    for row in range(1, raw.shape[0]):
        participant = _participant_number(raw.iat[row, 0])
        if participant is None:
            continue
        values[participant] = {}
        for bws, column in columns.items():
            value = pd.to_numeric(raw.iat[row, column], errors="coerce")
            values[participant][bws] = float(value) if pd.notna(value) else np.nan
    return values


def _metric_value(entry, key, channel):
    value = entry.get(key, {}).get(channel, np.nan)
    return float(value) if value is not None else np.nan


def _mean_if_available(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def emg_outcomes_table(data, participant=None):
    """Return raw, normalized, side-specific, and bilateral EMG outcomes."""
    rows = []
    participant_label = participant_tag(participant) if participant is not None else ""
    for direction, d in data.items():
        for bws, entry in d.get("emg", {}).items():
            channels_by_muscle = {}
            for channel in entry.get("env_window", {}):
                side, muscle = emg_processing.channel_side_and_muscle(channel)
                channels_by_muscle.setdefault(muscle, {})[side] = channel

            for muscle, side_channels in sorted(channels_by_muscle.items()):
                side_rows = {}
                for side, channel in side_channels.items():
                    if side not in {"L", "R"}:
                        continue
                    side_rows[side] = {
                        "participant": participant_label,
                        "direction": direction,
                        "bws_percent": bws,
                        "muscle": muscle,
                        "side": "left" if side == "L" else "right",
                        "channel": channel,
                        "mean_envelope_raw": _metric_value(entry, "mean_envelope", channel),
                        "rms_raw": _metric_value(entry, "rms", channel),
                        "iemg_raw": _metric_value(entry, "iemg", channel),
                        "mean_envelope_normalized": _metric_value(
                            entry, "mean_envelope_normalized", channel
                        ),
                        "rms_normalized": _metric_value(entry, "rms_normalized", channel),
                        "iemg_normalized": _metric_value(entry, "iemg_normalized", channel),
                        "envelope_normalization_peak": _metric_value(
                            entry, "normalization_peaks", channel
                        ),
                        "rms_normalization_reference": _metric_value(
                            entry, "rms_references", channel
                        ),
                        "window_duration_s": entry.get("window_duration_s", np.nan),
                        "unit": entry.get("unit") or "",
                        "file": entry.get("file", ""),
                    }
                    rows.append(side_rows[side])

                if "L" in side_rows and "R" in side_rows:
                    left, right = side_rows["L"], side_rows["R"]
                    bilateral = left.copy()
                    bilateral["side"] = "bilateral"
                    bilateral["channel"] = f"{left['channel']} / {right['channel']}"
                    for key in (
                        "mean_envelope_raw", "rms_raw", "iemg_raw",
                        "mean_envelope_normalized", "rms_normalized", "iemg_normalized",
                    ):
                        bilateral[key] = _mean_if_available([left[key], right[key]])
                    rows.append(bilateral)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["direction", "muscle", "bws_percent", "side"]
    )


def cci_table(data, participant=None):
    """Return side-specific and bilateral biceps-triceps CCI outcomes."""
    rows = []
    participant_label = participant_tag(participant) if participant is not None else ""
    for direction, d in data.items():
        for bws, entry in d.get("emg", {}).items():
            channels_by_muscle = {}
            for channel in entry.get("env_normalized_window", {}):
                side, muscle = emg_processing.channel_side_and_muscle(channel)
                channels_by_muscle.setdefault(muscle.lower(), {})[side] = channel

            biceps_key = next((key for key in channels_by_muscle if "biceps" in key), None)
            triceps_key = next((key for key in channels_by_muscle if "triceps" in key), None)
            if biceps_key is None or triceps_key is None:
                continue

            cci_values = {}
            for side in ("L", "R"):
                biceps_channel = channels_by_muscle[biceps_key].get(side)
                triceps_channel = channels_by_muscle[triceps_key].get(side)
                if biceps_channel is None or triceps_channel is None:
                    cci_values[side] = np.nan
                    continue
                cci_values[side] = emg_processing.calculate_cci(
                    entry["env_normalized_window"][biceps_channel],
                    entry["env_normalized_window"][triceps_channel],
                    float(entry["fs"]),
                )

            if not any(np.isfinite(value) for value in cci_values.values()):
                continue
            rows.append({
                "participant": participant_label,
                "direction": direction,
                "bws_percent": bws,
                "cci_left_percent": cci_values.get("L", np.nan),
                "cci_right_percent": cci_values.get("R", np.nan),
                "cci_bilateral_percent": _mean_if_available(list(cci_values.values())),
                "window_duration_s": entry.get("window_duration_s", np.nan),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["direction", "bws_percent"])


def force_outcomes_table(data, sync_report, force_dir, participant):
    """Calculate force outcomes from actual rope force in each climbing window."""
    nominal_values = load_nominal_force_values(force_dir)
    participant_values = nominal_values.get(int(participant), {})
    rows = []
    issues = []
    if sync_report is None or sync_report.empty:
        return pd.DataFrame(), ["No synchronization report available for force outcomes"]

    for direction, d in data.items():
        for bws, force in sorted(d.get("force", {}).items()):
            report_bws = pd.to_numeric(sync_report["bws_percent"], errors="coerce")
            matches = sync_report[(sync_report["direction"].astype(str) == str(direction)) &
                                  (report_bws == bws)]
            if matches.empty:
                issues.append(f"{direction} {bws}%: no synchronization-window row")
                continue
            row = matches.iloc[-1]
            try:
                start_s = float(row["analysis_start_s"])
                end_s = float(row["analysis_end_s"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{direction} {bws}%: invalid force-analysis window")
                continue
            if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
                issues.append(f"{direction} {bws}%: invalid force-analysis window")
                continue
            if cfg.FORCE_OUTCOME_COLUMN not in force["forces"]:
                issues.append(
                    f"{direction} {bws}%: force column '{cfg.FORCE_OUTCOME_COLUMN}' not found"
                )
                continue

            t = np.asarray(force["t"], dtype=float)
            # The eClimber export stores the measured-force values at one-tenth
            # of their physical Newton values, while the nominal lookup is
            # already converted to N. Keep raw data and timing unchanged, but
            # convert the measured samples before calculating all outcomes.
            measured = (
                force["forces"][cfg.FORCE_OUTCOME_COLUMN].to_numpy(dtype=float)
                * cfg.FORCE_MEASURED_SCALE
            )
            keep = np.isfinite(t) & np.isfinite(measured) & (t >= start_s) & (t <= end_s)
            values = measured[keep]
            if values.size == 0:
                issues.append(f"{direction} {bws}%: no measured force samples in window")
                continue

            target_kg = participant_values.get(bws, np.nan)
            nominal_force = (target_kg * cfg.NOMINAL_FORCE_KG_TO_N
                             if np.isfinite(target_kg) else np.nan)
            mean_force = float(np.mean(values))
            force_sd = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
            force_cv = (100.0 * force_sd / abs(mean_force)
                        if np.isfinite(force_sd) and mean_force != 0 else np.nan)
            ratio = (mean_force / nominal_force
                     if np.isfinite(nominal_force) and nominal_force != 0 else np.nan)
            rmse = (float(np.sqrt(np.mean((values - nominal_force) ** 2)))
                    if np.isfinite(nominal_force) else np.nan)
            sd_percent_nominal = (100.0 * force_sd / nominal_force
                                  if np.isfinite(force_sd) and np.isfinite(nominal_force)
                                  and nominal_force != 0 else np.nan)
            rows.append({
                "participant": participant_tag(participant),
                "direction": direction,
                "bws_percent": bws,
                "target_setting_kg": target_kg,
                "nominal_force_N": nominal_force,
                "mean_force_N": mean_force,
                "actual_to_nominal_ratio": ratio,
                "force_SD_N": force_sd,
                "force_SD_percent_nominal": sd_percent_nominal,
                "force_CV_percent": force_cv,
                "peak_force_N": float(np.max(values)),
                "force_RMSE_N": rmse,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "window_duration_s": end_s - start_s,
                "force_column": cfg.FORCE_OUTCOME_COLUMN,
                "force_file": force.get("file", ""),
            })

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["direction", "bws_percent"])
    return table, issues


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
    from . import config as cfg
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
