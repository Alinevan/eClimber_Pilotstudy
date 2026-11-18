#!/usr/bin/env python3
"""Interactive manual force/EMG synchronisation.

This is the required synchronization workflow: crop the derived signals,
select the EMG and force/position windows directly in the browser, detect the
three EMG peaks and three position dips in those selected windows, and inspect
the resulting shift before saving it to a report.

Usage:
    python -m emg_force.manual_synchronisation
    python -m emg_force.manual_synchronisation --data /path/to/data --participant 3

Stage 1 opens a condition-by-condition crop window. For every available
direction/BWS pair with both an EMG file and a force file, adjust the two
ranges and press "Save this condition and go to next"; pressing the button
with the full range selected still saves the full recording. Missing raw files
are reported but are not added to the workflow. After Stage 1 is closed,
Stage 2 opens using only those saved crop files and walks through the same
available pairs for manual peak/dip-window selection. All plots are
independently zoomable and the original recordings are never modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

from common.paths import output_subdirectory

from . import config as cfg
from . import emg_processing, io as data_loading, synchronisation
from .cli import ask_participant, get_data_folder


def _condition_values(data):
    """Return available paired conditions as ``direction|bws`` keys."""
    values = []
    for direction in cfg.DIRECTIONS:
        d = data.get(direction, {})
        levels = sorted(set(d.get("emg", {})) & set(d.get("force", {})))
        values.extend(f"{direction}|{bws}" for bws in levels)
    return values


def _parse_condition(value):
    direction, bws = value.split("|", 1)
    return direction, int(bws)


def _condition_label(value):
    direction, bws = _parse_condition(value)
    return f"{direction.capitalize()} — BWS {bws}%"


def _biceps_channels(entry):
    """Prefer the configured biceps labels, then any channel containing biceps."""
    available = list(entry.get("rectified", {}))
    preferred = [name for name in ("R.Biceps Br.", "L.Biceps Br.") if name in available]
    fallback = [name for name in available if "biceps" in name.lower() and name not in preferred]
    return preferred + fallback


def _preferred_force_channel(channels, detection_mode):
    """Choose a sensible default channel for the selected sync mode."""
    channels = list(channels)
    if detection_mode == "valleys":
        position = [c for c in channels if "position" in str(c).lower()]
        if position:
            return position[0]
    if cfg.SYNC_FORCE_COLUMN in channels:
        return cfg.SYNC_FORCE_COLUMN
    return channels[0] if channels else None


def _initial_crop_windows(data, condition):
    """Return full acquisition ranges for the active condition."""
    direction, bws = _parse_condition(condition)
    emg = data[direction]["emg"][bws]
    force = data[direction]["force"][bws]
    emg_t = np.asarray(emg["t"], dtype=float)
    force_t = np.asarray(force["t"], dtype=float)
    return {
        "emg": [float(emg_t[0]), float(emg_t[-1])] if len(emg_t) else [0.0, 0.0],
        "force": [float(force_t[0]), float(force_t[-1])] if len(force_t) else [0.0, 0.0],
    }


def _crop_mask(t, crop_window):
    """Boolean mask for a crop range while retaining the original time base."""
    if not crop_window:
        return np.ones(len(t), dtype=bool)
    start, end = sorted((float(crop_window[0]), float(crop_window[1])))
    return np.isfinite(t) & (t >= start) & (t <= end)


def _clamp_window(window, bounds):
    """Clamp a selectable window to the currently displayed crop."""
    start, end = sorted((float(bounds[0]), float(bounds[1])))
    if not window:
        return [start, end]
    selected_start, selected_end = sorted((float(window[0]), float(window[1])))
    selected_start = min(max(selected_start, start), end)
    selected_end = min(max(selected_end, start), end)
    return [min(selected_start, selected_end), max(selected_start, selected_end)]


def _save_cropped_files(data, condition, crop_windows, output_dir, participant):
    """Save rectified EMG channels and raw force values for one crop selection."""
    direction, bws = _parse_condition(condition)
    emg = data[direction]["emg"][bws]
    force = data[direction]["force"][bws]
    crop_dir = output_dir / "manual_crops" / f"{cfg.PARTICIPANT_PREFIX}{participant:02d}"
    crop_dir.mkdir(parents=True, exist_ok=True)

    emg_mask = _crop_mask(np.asarray(emg["t"], dtype=float), crop_windows["emg"])
    emg_out = pd.DataFrame({"time_s": np.asarray(emg["t"])[emg_mask]})
    for channel, signal in emg.get("rectified", {}).items():
        emg_out[channel] = np.asarray(signal)[emg_mask]
    emg_path = crop_dir / f"{direction}_BWS{bws}_rectified_emg.csv"
    emg_out.to_csv(emg_path, index=False)

    force_t = np.asarray(force["t"], dtype=float)
    force_mask = _crop_mask(force_t, crop_windows["force"])
    force_out = force["forces"].iloc[np.flatnonzero(force_mask)].copy().reset_index(drop=True)
    force_out.insert(0, "time_s", force_t[force_mask])
    force_path = crop_dir / f"{direction}_BWS{bws}_raw_force.csv"
    force_out.to_csv(force_path, index=False)
    return emg_path, force_path


def _crop_paths(output_dir, participant, condition):
    """Return the persisted EMG and force crop paths for one condition."""
    direction, bws = _parse_condition(condition)
    crop_dir = output_dir / "manual_crops" / f"{cfg.PARTICIPANT_PREFIX}{participant:02d}"
    return (
        crop_dir / f"{direction}_BWS{bws}_rectified_emg.csv",
        crop_dir / f"{direction}_BWS{bws}_raw_force.csv",
    )


def _saved_crop_windows(output_dir, participant, condition):
    """Read saved crop limits, or return ``None`` if the crop is unavailable."""
    emg_path, force_path = _crop_paths(output_dir, participant, condition)
    if not emg_path.exists() or not force_path.exists():
        return None
    try:
        emg_t = pd.read_csv(emg_path, usecols=["time_s"])["time_s"].to_numpy(dtype=float)
        force_t = pd.read_csv(force_path, usecols=["time_s"])["time_s"].to_numpy(dtype=float)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if len(emg_t) < 2 or len(force_t) < 2:
        return None
    if not (np.isfinite(emg_t[[0, -1]]).all() and np.isfinite(force_t[[0, -1]]).all()):
        return None
    return {
        "emg": [float(emg_t[0]), float(emg_t[-1])],
        "force": [float(force_t[0]), float(force_t[-1])],
    }


def all_crops_saved(data, output_dir, participant):
    """Return whether every available paired condition has a valid crop."""
    conditions = _condition_values(data)
    return bool(conditions) and all(
        _saved_crop_windows(output_dir, participant, condition) is not None
        for condition in conditions
    )


def _initial_emg_window(data, condition, crop_window=None):
    direction, bws = _parse_condition(condition)
    emg = data[direction]["emg"][bws]
    t = np.asarray(emg.get("t", []), dtype=float)
    if crop_window:
        start = float(crop_window[0])
        end = float(crop_window[1])
    elif len(t):
        start = float(t[0])
        end = float(t[-1])
    else:
        start = 0.0
        end = float(emg.get("duration", 0.0))
    return [start, min(end, start + cfg.SYNC_WINDOW_S)]


def _report_path(output_dir, participant):
    return output_dir / f"manual_sync_report_{cfg.PARTICIPANT_PREFIX}{participant:02d}.csv"


def _detection_window_report_path(output_dir, participant):
    return output_dir / f"manual_detection_windows_{cfg.PARTICIPANT_PREFIX}{participant:02d}.csv"


def _save_detection_windows(path, result):
    """Persist crop and peak/dip-window choices before running detection."""
    row = {
        key: value for key, value in result.items()
        if key not in {"emg_peaks_s", "force_peaks_s", "synchronised"}
    }
    row["window_selection_saved"] = True
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old["direction"] == result["direction"]) &
                    (old["bws_percent"] == result["bws_percent"]))]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def _save_report(path, result):
    row = {key: value for key, value in result.items()
           if key not in {"emg_peaks_s", "force_peaks_s"}}
    row["emg_peaks_s"] = json.dumps(result["emg_peaks_s"])
    row["force_peaks_s"] = json.dumps(result["force_peaks_s"])
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old["direction"] == result["direction"]) &
                    (old["bws_percent"] == result["bws_percent"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def _saved_window_selection(path, condition):
    """Load a saved peak/dip-window selection without requiring a sync result."""
    if not path.exists():
        return None
    try:
        saved = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
    direction, bws = _parse_condition(condition)
    if not {"direction", "bws_percent"}.issubset(saved.columns):
        return None
    matches = saved[(saved["direction"].astype(str) == direction) &
                    (pd.to_numeric(saved["bws_percent"], errors="coerce") == bws)]
    if matches.empty:
        return None
    row = matches.iloc[-1]
    result = {}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _saved_sync_result(path, condition):
    """Load one previously saved synchronization result for a condition."""
    if not path.exists():
        return None
    try:
        saved = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
    direction, bws = _parse_condition(condition)
    if not {"direction", "bws_percent"}.issubset(saved.columns):
        return None
    matches = saved[(saved["direction"].astype(str) == direction) &
                    (pd.to_numeric(saved["bws_percent"], errors="coerce") == bws)]
    if matches.empty:
        return None
    row = matches.iloc[-1]
    result = {}
    list_fields = {"emg_peaks_s", "force_peaks_s", "position_dips_s"}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if key in list_fields and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _window_selection_result(data, condition, emg_channel, force_channel,
                             emg_window, force_window):
    """Build the standalone record for one selected detection-window pair."""
    direction, bws = _parse_condition(condition)
    emg = data[direction]["emg"][bws]
    force = data[direction]["force"][bws]
    return {
        "direction": direction,
        "bws_percent": bws,
        "force_file": force["file"],
        "emg_file": emg["file"],
        "emg_channel": emg_channel,
        "force_channel": force_channel,
        "force_window_start_s": float(force_window[0]),
        "force_window_end_s": float(force_window[1]),
        "emg_window_start_s": float(emg_window[0]),
        "emg_window_end_s": float(emg_window[1]),
        "emg_crop_start_s": float(emg["t"][0]),
        "emg_crop_end_s": float(emg["t"][-1]),
        "force_crop_start_s": float(force["t"][0]),
        "force_crop_end_s": float(force["t"][-1]),
        "detection_mode": "valleys",
        "position_channel": force_channel,
    }


def _window_selection_matches(saved, emg_channel, force_channel,
                              emg_window, force_window):
    """Check that the sliders still match the separately saved selection."""
    if saved is None or saved.get("emg_channel") != emg_channel or saved.get("force_channel") != force_channel:
        return False
    for key, window in (
        (("emg_window_start_s", "emg_window_end_s"), emg_window),
        (("force_window_start_s", "force_window_end_s"), force_window),
    ):
        if not all(name in saved for name in key):
            return False
        if not np.allclose(
            [float(saved[key[0]]), float(saved[key[1]])],
            [float(window[0]), float(window[1])],
        ):
            return False
    return True


def _detect_sync_result(data, condition, detection_mode, force_channel,
                        emg_window, force_window, emg_channel=None):
    """Detect the manually requested events and build one report row."""
    direction, bws = _parse_condition(condition)
    emg = data[direction]["emg"][bws]
    force = data[direction]["force"][bws]
    if force_channel not in force["forces"]:
        return None, f"Force/position channel not found: {force_channel}"

    biceps = _biceps_channels(emg)
    emg_label = emg_channel if emg_channel in emg["rectified"] else (
        cfg.SYNC_EMG_LABEL if cfg.SYNC_EMG_LABEL in emg["rectified"]
        else (biceps[0] if biceps else None)
    )
    if emg_label is None:
        return None, "No EMG synchronization channel is available."

    emg_peaks = synchronisation.find_emg_sync_peaks_in_window(
        emg["t"], emg["rectified"][emg_label], emg_window[0], emg_window[1],
        lp_hz=cfg.ENVELOPE_LP_HZ,
    )
    force_invert = detection_mode == "valleys" or cfg.SYNC_FORCE_INVERT
    force_peaks = synchronisation.find_sync_peaks_in_window(
        force["t"], force["forces"][force_channel].to_numpy(),
        force_window[0], force_window[1], cfg.SYNC_FORCE_LP_HZ, force_invert,
    )
    if emg_peaks is None or force_peaks is None:
        missing = []
        if emg_peaks is None:
            missing.append(f"fewer than {cfg.SYNC_N_PEAKS} EMG peaks")
        if force_peaks is None:
            missing.append(f"fewer than {cfg.SYNC_N_PEAKS} force events")
        return None, "Could not synchronize: " + "; ".join(missing) + "."

    # The requested synchronization reference is the last EMG peak and the
    # last position valley. The spread is retained as a quality indicator for
    # the other two event pairs.
    offset = float(emg_peaks[-1] - force_peaks[-1])
    spread = float(np.ptp(emg_peaks - force_peaks))
    analysis_window = (synchronisation.suggest_analysis_window(
        direction, force["t"], force["forces"][force_channel].to_numpy(),
        force_peaks, offset_s=offset,
    ) if detection_mode == "valleys" else None)
    result = {
        "direction": direction,
        "bws_percent": bws,
        "force_file": force["file"],
        "emg_file": emg["file"],
        "emg_channel": emg_label,
        "force_channel": force_channel,
        "force_window_start_s": force_window[0],
        "force_window_end_s": force_window[1],
        "emg_window_start_s": emg_window[0],
        "emg_window_end_s": emg_window[1],
        "emg_crop_start_s": float(emg["t"][0]),
        "emg_crop_end_s": float(emg["t"][-1]),
        "force_crop_start_s": float(force["t"][0]),
        "force_crop_end_s": float(force["t"][-1]),
        "detection_mode": detection_mode,
        "offset_s": offset,
        "peak_offset_spread_s": spread,
        "emg_peaks_s": np.round(emg_peaks, 6).tolist(),
        "force_peaks_s": np.round(force_peaks, 6).tolist(),
        "position_channel": force_channel if detection_mode == "valleys" else None,
        "position_dips_s": np.round(force_peaks, 6).tolist() if detection_mode == "valleys" else None,
        "synchronised": True,
    }
    if analysis_window:
        result.update(analysis_window)
    return result, None


def _make_stage_figure(data, condition, stage, crop_window=None,
                       detection_window=None, force_channel=None,
                       sync_result=None):
    """Create one independently zoomable EMG or force stage figure."""
    direction, bws = _parse_condition(condition)
    emg = data[direction]["emg"][bws]
    force = data[direction]["force"][bws]
    if stage == "emg":
        channels = list(emg.get("rectified", {}))
        t = np.asarray(emg["t"], dtype=float)
        titles = [f"Rectified EMG: {cfg.EMG_DISPLAY_LABELS.get(channel, channel)}"
                  for channel in channels]
        source = {channel: np.asarray(emg["rectified"][channel]) for channel in channels}
        axis_title = "Rectified EMG"
    else:
        channels = list(force["forces"].columns)
        offset = float(sync_result.get("offset_s", 0.0)) if sync_result else 0.0
        t = np.asarray(force["t"], dtype=float) + offset
        titles = [f"Raw force: {channel}" for channel in channels]
        source = {channel: force["forces"][channel].to_numpy() for channel in channels}
        axis_title = "Raw force / position"

    if not channels:
        titles = ["No channels available"]
        source = {}

    fig = make_subplots(rows=len(titles), cols=1, shared_xaxes=False,
                        subplot_titles=titles,
                        vertical_spacing=min(0.08, 0.7 / len(titles)))
    for row, channel in enumerate(channels, start=1):
        y = source[channel]
        n = min(len(t), len(y))
        fig.add_trace(go.Scattergl(x=t[:n], y=y[:n], mode="lines",
                                   name=channel, legendgroup=channel,
                                   line=dict(width=1.1)), row=row, col=1)

    def shifted_window(window):
        if not window:
            return None
        offset = float(sync_result.get("offset_s", 0.0)) if sync_result and stage == "force" else 0.0
        return [float(window[0]) + offset, float(window[1]) + offset]

    crop_display = shifted_window(crop_window)
    detection_display = shifted_window(detection_window)
    if crop_display:
        for row in range(1, len(titles) + 1):
            fig.add_vrect(x0=crop_display[0], x1=crop_display[1],
                          fillcolor="royalblue", opacity=0.08, line_width=1,
                          line_dash="dash", layer="below", row=row, col=1)
    if detection_display:
        for row in range(1, len(titles) + 1):
            fig.add_vrect(x0=detection_display[0], x1=detection_display[1],
                          fillcolor="orange", opacity=0.12, line_width=1,
                          line_dash="dot", layer="below", row=row, col=1)

    emg_marker_overlay = None
    if sync_result:
        if stage == "emg":
            selected_channel = sync_result.get("emg_channel")
            if selected_channel in channels:
                selected_row = channels.index(selected_channel) + 1
                peaks = np.asarray(sync_result.get("emg_peaks_s", []), dtype=float)
                raw_t = np.asarray(t, dtype=float)
                raw_y = np.asarray(source[selected_channel], dtype=float)
                n = min(len(raw_t), len(raw_y))
                raw_t, raw_y = raw_t[:n], raw_y[:n]
                if detection_window:
                    detection_mask = (
                        np.isfinite(raw_t) & np.isfinite(raw_y)
                        & (raw_t >= float(detection_window[0]))
                        & (raw_t <= float(detection_window[1]))
                    )
                    detection_input_t = raw_t[detection_mask]
                    detection_input_y = raw_y[detection_mask]
                else:
                    # Do not create a full-recording detection signal when no
                    # manual detection window is available yet.
                    detection_input_t = np.array([], dtype=float)
                    detection_input_y = np.array([], dtype=float)
                detection_t, detection_y = synchronisation.emg_sync_signal(
                    detection_input_t, detection_input_y, lp_hz=cfg.ENVELOPE_LP_HZ
                )
                if len(detection_t):
                    fig.add_trace(
                        go.Scattergl(x=detection_t, y=detection_y, mode="lines",
                                     name="EMG peak-detection signal",
                                     line=dict(color=cfg.SYNC_EMG_DETECTION_COLOR, width=1.8),
                                     legendgroup="sync-events"),
                        row=selected_row, col=1,
                    )
                    valid = (np.isfinite(peaks)
                             & (peaks >= detection_t[0])
                             & (peaks <= detection_t[-1]))
                    marker_t = peaks[valid]
                    marker_y = np.interp(marker_t, detection_t, detection_y)
                    emg_marker_overlay = (selected_row, marker_t, marker_y)
        else:
            selected_channel = sync_result.get("position_channel") or force_channel
            if selected_channel in channels:
                selected_row = channels.index(selected_channel) + 1
                offset = float(sync_result.get("offset_s", 0.0))
                dips = np.asarray(sync_result.get("force_peaks_s", []), dtype=float) + offset
                y = source[selected_channel]
                valid = np.isfinite(dips) & (dips >= t[0]) & (dips <= t[-1])
                fig.add_trace(
                    go.Scatter(x=dips[valid], y=np.interp(dips[valid], t, y),
                               mode="markers", name="Detected position dips (3)",
                               marker=dict(color=cfg.SYNC_POSITION_PEAK_COLOR, size=12, symbol="x"),
                               legendgroup="sync-events",
                               hovertemplate="Position valley %{x:.3f} s<extra></extra>"),
                    row=selected_row, col=1,
                )
            if (sync_result.get("analysis_start_s") is not None
                    and sync_result.get("analysis_end_s") is not None):
                for row in range(1, len(titles) + 1):
                    fig.add_vrect(x0=sync_result["analysis_start_s"],
                                  x1=sync_result["analysis_end_s"],
                                  fillcolor="seagreen", opacity=0.12,
                                  line_color="seagreen", line_width=1,
                                  line_dash="dash", layer="below", row=row, col=1,
                                  annotation_text="calculation window" if row == 1 else None,
                                  annotation_position="top left" if row == 1 else None)
        if (stage == "emg"
                and sync_result.get("analysis_start_s") is not None
                and sync_result.get("analysis_end_s") is not None):
            for row in range(1, len(titles) + 1):
                fig.add_vrect(x0=sync_result["analysis_start_s"],
                              x1=sync_result["analysis_end_s"],
                              fillcolor="seagreen", opacity=0.12,
                              line_color="seagreen", line_width=1,
                              line_dash="dash", layer="below", row=row, col=1,
                              annotation_text="calculation window" if row == 1 else None,
                              annotation_position="top left" if row == 1 else None)

    # Add the EMG markers after every signal and shaded region so they remain
    # visibly in the foreground of the selected biceps subplot.
    if emg_marker_overlay is not None:
        selected_row, marker_t, marker_y = emg_marker_overlay
        fig.add_trace(
            go.Scattergl(
                x=marker_t, y=marker_y, mode="markers",
                name="Detected EMG peaks (3)",
                marker=dict(color=cfg.SYNC_EMG_PEAK_COLOR, size=12, symbol="x"),
                legendgroup="sync-events",
                hovertemplate="EMG peak %{x:.3f} s<extra></extra>",
            ),
            row=selected_row, col=1,
        )

    fig.update_layout(
        title=(f"{_condition_label(condition)} — "
               f"{'processed EMG' if stage == 'emg' else 'raw force/position'}"
               + (f" — force shifted {float(sync_result.get('offset_s', 0.0)):+.3f} s"
                  if sync_result and stage == "force" else "")),
        template="plotly_white",
        height=max(420, 145 * len(titles)),
        margin=dict(t=70, r=30, b=55, l=70),
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(groupclick="togglegroup"),
    )
    fig.update_xaxes(title_text="Time (s)", row=len(titles), col=1)
    fig.update_yaxes(title_text=axis_title, row=1, col=1)
    if crop_display:
        for row in range(1, len(titles) + 1):
            fig.update_xaxes(range=crop_display, row=row, col=1)
    return fig


def _entry_time_bounds(entry):
    """Return the first and last valid time values of a loaded entry."""
    t = np.asarray(entry.get("t", []), dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return [0.0, 1.0]
    return [float(t[0]), float(t[-1])]


def _condition_crop_bounds(data, condition):
    direction, bws = _parse_condition(condition)
    return {
        "emg": _entry_time_bounds(data[direction]["emg"][bws]),
        "force": _entry_time_bounds(data[direction]["force"][bws]),
    }


def _calculation_end_slider_state(data, condition, preview):
    """Return the endpoint-control range in synchronized graph time."""
    direction, _bws = _parse_condition(condition)
    crop = _condition_crop_bounds(data, condition)
    offset = 0.0
    if preview is not None:
        try:
            offset = float(preview.get("offset_s", 0.0))
        except (TypeError, ValueError):
            offset = 0.0
    low = float(crop["force"][0]) + offset
    high = float(crop["force"][1]) + offset
    high = max(high, low + 1.0)

    if preview is None or direction != "horizontal":
        return low, high, high, True

    try:
        start = float(preview["analysis_start_s"])
        end = float(preview["analysis_end_s"])
    except (KeyError, TypeError, ValueError):
        return low, high, high, True
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return low, high, high, True

    low = max(low, start)
    high = max(high, end, low + 0.01)
    return low, high, min(max(end, low), high), False


def _crop_stage_state(data, condition, output_dir, participant):
    """Return crop ranges and a short note for the active condition."""
    saved = _saved_crop_windows(output_dir, participant, condition)
    if saved is not None:
        return saved, "A saved crop was loaded. Adjust it if needed, then save again."
    return _condition_crop_bounds(data, condition), "No saved crop yet. The full recording is selected."


def create_crop_app(data, participant, output_dir):
    """Create the sequential Stage 1 application for saving every crop."""
    conditions = _condition_values(data)
    if not conditions:
        raise SystemExit("No conditions contain both an EMG file and a force file.")

    first = conditions[0]
    first_crop, _first_note = _crop_stage_state(data, first, output_dir, participant)
    app = Dash(__name__)
    app.title = "Stage 1 — Save EMG and force crops"
    app.layout = html.Div([
        html.H2("Stage 1 of 2 — Save the crop for every available condition"),
        html.P(
            "Work from top to bottom. For each condition, adjust the EMG and "
            "force/position ranges if needed, then press the blue button. "
            "Pressing it saves both files immediately and advances to the next condition."
        ),
        html.Div(id="crop-progress", style={"fontWeight": "bold", "marginBottom": "4px"}),
        html.H3(id="crop-condition"),
        html.Div(id="crop-load-note", style={"marginBottom": "12px"}),
        html.H4("1A. EMG crop"),
        dcc.Graph(
            id="crop-emg-graph",
            figure=_make_stage_figure(data, first, "emg", crop_window=first_crop["emg"]),
            config={"displaylogo": False, "scrollZoom": True},
        ),
        html.Label("EMG crop window — select the complete EMG interval to keep"),
        dcc.RangeSlider(
            id="emg-crop-window",
            min=first_crop["emg"][0], max=max(first_crop["emg"][1], first_crop["emg"][0] + 1.0),
            value=first_crop["emg"], allowCross=False,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.H4("1B. Force and position crop"),
        dcc.Graph(
            id="crop-force-graph",
            figure=_make_stage_figure(data, first, "force", crop_window=first_crop["force"]),
            config={"displaylogo": False, "scrollZoom": True},
        ),
        html.Label("Force/position crop window — select the complete force interval to keep"),
        dcc.RangeSlider(
            id="force-crop-window",
            min=first_crop["force"][0], max=max(first_crop["force"][1], first_crop["force"][0] + 1.0),
            value=first_crop["force"], allowCross=False,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div([
            html.Button("Previous condition", id="crop-previous", n_clicks=0),
            html.Button("Save this condition and go to next", id="crop-save-next", n_clicks=0,
                        style={"backgroundColor": "#1976d2", "color": "white"}),
        ], style={"display": "flex", "gap": "10px", "margin": "16px 0"}),
        html.Div(id="crop-status", style={"whiteSpace": "pre-wrap", "fontWeight": "bold"}),
        dcc.Store(id="crop-index", data=0),
    ])

    @app.callback(
        Output("crop-condition", "children"),
        Output("crop-progress", "children"),
        Output("crop-load-note", "children"),
        Output("emg-crop-window", "min"),
        Output("emg-crop-window", "max"),
        Output("emg-crop-window", "value"),
        Output("force-crop-window", "min"),
        Output("force-crop-window", "max"),
        Output("force-crop-window", "value"),
        Output("crop-save-next", "children"),
        Output("crop-previous", "disabled"),
        Input("crop-index", "data"),
    )
    def render_crop_stage(index):
        index = min(max(int(index or 0), 0), len(conditions) - 1)
        condition = conditions[index]
        crop, note = _crop_stage_state(data, condition, output_dir, participant)
        return (
            _condition_label(condition),
            f"Available condition {index + 1} of {len(conditions)} — save each before continuing",
            note,
            crop["emg"][0], max(crop["emg"][1], crop["emg"][0] + 1.0), crop["emg"],
            crop["force"][0], max(crop["force"][1], crop["force"][0] + 1.0), crop["force"],
            "Save this condition and finish cropping" if index == len(conditions) - 1
            else "Save this condition and go to next",
            index == 0,
        )

    @app.callback(
        Output("crop-emg-graph", "figure"),
        Output("crop-force-graph", "figure"),
        Input("crop-index", "data"),
        Input("emg-crop-window", "value"),
        Input("force-crop-window", "value"),
    )
    def redraw_crop_stage(index, emg_crop, force_crop):
        condition = conditions[min(max(int(index or 0), 0), len(conditions) - 1)]
        return (
            _make_stage_figure(data, condition, "emg", crop_window=emg_crop),
            _make_stage_figure(data, condition, "force", crop_window=force_crop),
        )

    @app.callback(
        Output("crop-index", "data"),
        Output("crop-status", "children"),
        Input("crop-save-next", "n_clicks"),
        Input("crop-previous", "n_clicks"),
        State("crop-index", "data"),
        State("emg-crop-window", "value"),
        State("force-crop-window", "value"),
        prevent_initial_call=True,
    )
    def navigate_crop(_save_clicks, _previous_clicks, index, emg_crop, force_crop):
        index = min(max(int(index or 0), 0), len(conditions) - 1)
        if ctx.triggered_id == "crop-previous":
            new_index = max(0, index - 1)
            return new_index, "Moved back. The saved crop will be loaded for that condition."

        condition = conditions[index]
        _save_cropped_files(
            data, condition, {"emg": emg_crop, "force": force_crop}, output_dir, participant
        )
        if index < len(conditions) - 1:
            return (
                index + 1,
                f"Saved {_condition_label(condition)}. Now continue with the next condition.",
            )
        return (
            index,
            "Stage 1 complete. To continue to Stage 2, press Ctrl+C in the terminal. "
            "Stage 2 will open automatically.",
        )

    return app


def _sync_stage_state(data, condition, detection_report_path, sync_report_path):
    """Return defaults for a cropped condition and saved checkpoint data."""
    direction, bws = _parse_condition(condition)
    emg = data[direction]["emg"][bws]
    force = data[direction]["force"][bws]
    crop = _condition_crop_bounds(data, condition)
    saved = _saved_sync_result(sync_report_path, condition)
    saved_windows = _saved_window_selection(detection_report_path, condition)
    window_source = saved_windows or saved
    mode = "valleys"
    emg_channels = _biceps_channels(emg)
    emg_channel = window_source.get("emg_channel") if window_source else None
    if emg_channel not in emg["rectified"]:
        emg_channel = (
            cfg.SYNC_EMG_LABEL if cfg.SYNC_EMG_LABEL in emg["rectified"]
            else (emg_channels[0] if emg_channels else None)
        )
    channels = list(force["forces"].columns)
    force_channel = window_source.get("force_channel") if window_source else None
    if force_channel not in channels:
        force_channel = _preferred_force_channel(channels, mode)
    emg_window = [
        float(window_source["emg_window_start_s"]), float(window_source["emg_window_end_s"])
    ] if window_source and all(key in window_source for key in ("emg_window_start_s", "emg_window_end_s")) else _initial_emg_window(data, condition, crop["emg"])
    force_window = [
        float(window_source["force_window_start_s"]), float(window_source["force_window_end_s"])
    ] if window_source and all(key in window_source for key in ("force_window_start_s", "force_window_end_s")) else crop["force"]
    return (
        crop,
        mode,
        emg_channel,
        force_channel,
        _clamp_window(emg_window, crop["emg"]),
        _clamp_window(force_window, crop["force"]),
        saved,
        saved_windows,
    )


def create_sync_app(data, participant, output_dir):
    """Create the sequential Stage 2 application using saved crop files."""
    conditions = _condition_values(data)
    if not conditions:
        raise SystemExit("No cropped EMG/force conditions are available.")
    report_path = _report_path(output_dir, participant)
    detection_report_path = _detection_window_report_path(output_dir, participant)
    first = conditions[0]
    first_crop, _mode, first_emg_channel, first_force_channel, first_emg_window, first_force_window, first_result, _first_windows = _sync_stage_state(
        data, first, detection_report_path, report_path
    )
    direction, bws = _parse_condition(first)
    first_channels = list(data[direction]["force"][bws]["forces"].columns)

    app = Dash(__name__)
    app.title = "Stage 2 — Manual synchronization"
    app.layout = html.Div([
        html.H2("Stage 2 of 2 — Select synchronization windows"),
        html.P(
            "The data shown here are the saved crop files from Stage 1. "
            "For each condition, select the EMG window containing the three "
            "biceps peaks and the force/position window containing the three dips. "
            "First save the peak/dip windows. Then run detection and synchronization."
        ),
        html.Div(id="sync-progress", style={"fontWeight": "bold", "marginBottom": "4px"}),
        html.H3(id="sync-condition"),
        html.Div(id="sync-load-note", style={"marginBottom": "12px"}),
        html.Label("Biceps EMG channel used for the three peaks"),
        dcc.Dropdown(
            id="sync-emg-channel",
            options=[{"label": channel, "value": channel} for channel in _biceps_channels(data[direction]["emg"][bws])],
            value=first_emg_channel,
            clearable=False,
        ),
        html.Label("Force/position channel used for the three dips"),
        dcc.Dropdown(
            id="sync-force-channel",
            options=[{"label": channel, "value": channel} for channel in first_channels],
            value=first_force_channel,
            clearable=False,
        ),
        html.H4("2A. EMG peak-detection window"),
        dcc.Graph(
            id="sync-emg-graph",
            figure=_make_stage_figure(data, first, "emg", crop_window=first_crop["emg"],
                                      detection_window=first_emg_window, sync_result=first_result),
            config={"displaylogo": False, "scrollZoom": True},
        ),
        html.Label("EMG peak-detection window — select the range containing the three peaks"),
        dcc.RangeSlider(
            id="sync-emg-window",
            min=first_crop["emg"][0], max=max(first_crop["emg"][1], first_crop["emg"][0] + 1.0),
            value=first_emg_window, allowCross=False,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.H4("2B. Force position-dip detection window"),
        dcc.Graph(
            id="sync-force-graph",
            figure=_make_stage_figure(data, first, "force", crop_window=first_crop["force"],
                                      detection_window=first_force_window, force_channel=first_force_channel,
                                      sync_result=first_result),
            config={"displaylogo": False, "scrollZoom": True},
        ),
        html.P(
            "After synchronization, the green shading shows the calculation window. "
            "For horizontal trials, click the desired endpoint on the position plot "
            "or adjust the endpoint slider below to replace only the end; the "
            "automatically calculated start remains fixed.",
            style={"marginTop": "-4px", "color": "#444"},
        ),
        html.Label("Horizontal calculation-window end (after synchronization)"),
        dcc.Slider(
            id="sync-calculation-end",
            min=first_crop["force"][0],
            max=max(first_crop["force"][1], first_crop["force"][0] + 1.0),
            value=first_crop["force"][1],
            step=0.01,
            disabled=True,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("Force/position dip-detection window — select the range containing the three dips"),
        dcc.RangeSlider(
            id="sync-force-window",
            min=first_crop["force"][0], max=max(first_crop["force"][1], first_crop["force"][0] + 1.0),
            value=first_force_window, allowCross=False,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("Automatic force/position alignment shift (seconds)"),
        dcc.Input(
            id="sync-offset-value",
            type="number",
            value=0.0,
            step=0.001,
            disabled=True,
            style={"width": "140px", "marginTop": "4px"},
        ),
        html.Div([
            html.Button("Previous condition", id="sync-previous", n_clicks=0),
            html.Button("Save peak/dip windows", id="sync-save-windows", n_clicks=0),
            html.Button("Detect peaks/dips and show alignment", id="sync-preview", n_clicks=0),
        ], style={"display": "flex", "gap": "10px", "margin": "16px 0"}),
        html.Div(id="sync-preview-status", style={"whiteSpace": "pre-wrap", "marginBottom": "8px"}),
        html.Button("Save synchronization and go to next", id="sync-run-next", n_clicks=0,
                    disabled=first_result is None,
                    style={"backgroundColor": "#1976d2", "color": "white", "margin": "16px 0"}),
        html.Div(id="sync-status", style={"whiteSpace": "pre-wrap", "fontWeight": "bold"}),
        dcc.Store(id="sync-index", data=0),
        dcc.Store(id="sync-preview-result", data=first_result),
    ])

    @app.callback(
        Output("sync-condition", "children"),
        Output("sync-progress", "children"),
        Output("sync-load-note", "children"),
        Output("sync-emg-channel", "options"),
        Output("sync-emg-channel", "value"),
        Output("sync-force-channel", "options"),
        Output("sync-force-channel", "value"),
        Output("sync-emg-window", "min"),
        Output("sync-emg-window", "max"),
        Output("sync-emg-window", "value"),
        Output("sync-force-window", "min"),
        Output("sync-force-window", "max"),
        Output("sync-force-window", "value"),
        Output("sync-run-next", "children"),
        Output("sync-previous", "disabled"),
        Input("sync-index", "data"),
    )
    def render_sync_stage(index):
        index = min(max(int(index or 0), 0), len(conditions) - 1)
        condition = conditions[index]
        crop, _mode, emg_channel, force_channel, emg_window, force_window, saved, saved_windows = _sync_stage_state(
            data, condition, detection_report_path, report_path
        )
        direction, bws = _parse_condition(condition)
        emg_channels = _biceps_channels(data[direction]["emg"][bws])
        channels = list(data[direction]["force"][bws]["forces"].columns)
        if saved:
            note = "A saved synchronization result was loaded. Adjust the windows and save again if needed."
        elif saved_windows:
            note = "Saved peak/dip windows were loaded. Adjust them if needed, then save again."
        else:
            note = "No peak/dip windows saved yet for this condition."
        return (
            _condition_label(condition),
            f"Available condition {index + 1} of {len(conditions)} — save each synchronization before continuing",
            note,
            [{"label": channel, "value": channel} for channel in emg_channels], emg_channel,
            [{"label": channel, "value": channel} for channel in channels], force_channel,
            crop["emg"][0], max(crop["emg"][1], crop["emg"][0] + 1.0), emg_window,
            crop["force"][0], max(crop["force"][1], crop["force"][0] + 1.0), force_window,
            "Detect peaks/dips, synchronize, and finish" if index == len(conditions) - 1
            else "Detect peaks/dips, synchronize, and go to next",
            index == 0,
        )

    @app.callback(
        Output("sync-emg-graph", "figure"),
        Output("sync-force-graph", "figure"),
        Input("sync-index", "data"),
        Input("sync-emg-window", "value"),
        Input("sync-force-window", "value"),
        Input("sync-force-channel", "value"),
        Input("sync-preview-result", "data"),
    )
    def redraw_sync_stage(index, emg_window, force_window, force_channel, sync_result):
        condition = conditions[min(max(int(index or 0), 0), len(conditions) - 1)]
        crop = _condition_crop_bounds(data, condition)
        return (
            _make_stage_figure(data, condition, "emg", crop_window=crop["emg"],
                               detection_window=emg_window, sync_result=sync_result),
            _make_stage_figure(data, condition, "force", crop_window=crop["force"],
                               detection_window=force_window, force_channel=force_channel,
                               sync_result=sync_result),
        )

    @app.callback(
        Output("sync-preview-result", "data"),
        Output("sync-preview-status", "children"),
        Output("sync-offset-value", "value"),
        Output("sync-run-next", "disabled"),
        Output("sync-calculation-end", "min"),
        Output("sync-calculation-end", "max"),
        Output("sync-calculation-end", "value"),
        Output("sync-calculation-end", "disabled"),
        Input("sync-index", "data"),
        Input("sync-save-windows", "n_clicks"),
        Input("sync-preview", "n_clicks"),
        State("sync-emg-channel", "value"),
        State("sync-force-channel", "value"),
        State("sync-emg-window", "value"),
        State("sync-force-window", "value"),
        prevent_initial_call=False,
    )
    def prepare_sync_preview(index, _save_window_clicks, _preview_clicks,
                             emg_channel, force_channel, emg_window, force_window):
        index = min(max(int(index or 0), 0), len(conditions) - 1)
        condition = conditions[index]

        if ctx.triggered_id == "sync-index" or ctx.triggered_id is None:
            saved = _saved_sync_result(report_path, condition)
            if saved:
                end_controls = _calculation_end_slider_state(data, condition, saved)
                return (saved, "Saved synchronization loaded. The automatic shift is shown below.",
                        float(saved.get("offset_s", 0.0)), False,
                        *end_controls)
            end_controls = _calculation_end_slider_state(data, condition, None)
            return (None, "Save the peak/dip windows, then detect to preview the alignment.",
                    0.0, True, *end_controls)

        saved_windows = _saved_window_selection(detection_report_path, condition)
        if ctx.triggered_id == "sync-save-windows":
            selection = _window_selection_result(
                data, condition, emg_channel, force_channel, emg_window, force_window
            )
            _save_detection_windows(detection_report_path, selection)
            end_controls = _calculation_end_slider_state(data, condition, None)
            return (
                None,
                f"Saved the peak/dip windows for {_condition_label(condition)}. "
                "Now press Detect peaks/dips and show alignment.",
                0.0, True, *end_controls,
            )

        if saved_windows is None:
            end_controls = _calculation_end_slider_state(data, condition, None)
            return (None, "First press 'Save peak/dip windows'.", 0.0,
                    True, *end_controls)
        if not _window_selection_matches(saved_windows, emg_channel, force_channel,
                                         emg_window, force_window):
            end_controls = _calculation_end_slider_state(data, condition, None)
            return (None, "The selected windows changed after the last save. Save the windows again.",
                    0.0, True, *end_controls)

        result, error = _detect_sync_result(
            data, condition, "valleys", force_channel, emg_window, force_window,
            emg_channel=emg_channel,
        )
        if error:
            end_controls = _calculation_end_slider_state(data, condition, None)
            return (None, error + " Adjust the windows and try again.", 0.0,
                    True, *end_controls)
        direction, _bws = _parse_condition(condition)
        if direction == "horizontal":
            if result.get("analysis_return_detected"):
                window_note = (
                    f"Calculated window: {result['analysis_start_s']:.3f}–"
                    f"{result['analysis_end_s']:.3f} s. Click the position plot "
                    "to replace only the endpoint if needed."
                )
            else:
                window_note = (
                    f"Start fixed at {result['analysis_start_s']:.3f} s, but no "
                    "sustained return was detected automatically. Click the "
                    "position plot to select the endpoint."
                )
        else:
            window_note = (
                f"Calculated window: {result['analysis_start_s']:.3f}–"
                f"{result['analysis_end_s']:.3f} s."
            )
        end_controls = _calculation_end_slider_state(data, condition, result)
        return (
            result,
            f"Detected 3 EMG peaks on {emg_channel}: {np.round(result['emg_peaks_s'], 3).tolist()} s\n"
            f"Detected 3 position valleys on {force_channel}: {np.round(result['force_peaks_s'], 3).tolist()} s\n"
            f"Automatic force shift: {result['offset_s']:+.3f} s. Inspect the crosses, then save.\n"
            f"{window_note}",
            float(result["offset_s"]), False, *end_controls,
        )

    @app.callback(
        Output("sync-preview-result", "data", allow_duplicate=True),
        Output("sync-preview-status", "children", allow_duplicate=True),
        Output("sync-calculation-end", "value", allow_duplicate=True),
        Input("sync-force-graph", "clickData"),
        State("sync-preview-result", "data"),
        State("sync-index", "data"),
        prevent_initial_call=True,
    )
    def select_horizontal_calculation_end(click_data, preview, index):
        """Use a clicked synchronized-position time as a horizontal endpoint."""
        if not click_data or preview is None:
            return no_update, no_update, no_update
        condition = conditions[min(max(int(index or 0), 0), len(conditions) - 1)]
        direction, _bws = _parse_condition(condition)
        if direction != "horizontal":
            return (no_update,
                    "Manual calculation-end selection is only available for horizontal trials.",
                    no_update)

        points = click_data.get("points") or []
        if not points or points[0].get("x") is None:
            return no_update, no_update, no_update
        try:
            selected_end = float(points[0]["x"])
            start = float(preview["analysis_start_s"])
        except (KeyError, TypeError, ValueError):
            return (no_update,
                    "Synchronize the condition first, then click the position plot.",
                    no_update)
        if selected_end <= start:
            return (no_update,
                    f"The endpoint must be after the fixed start at {start:.3f} s. "
                    "Click farther to the right on the position plot.",
                    no_update)

        adjusted = dict(preview)
        adjusted["analysis_end_s"] = selected_end
        adjusted["analysis_window_method"] = (
            "horizontal: manual endpoint selected on synchronized position plot"
        )
        adjusted["analysis_return_detected"] = True
        return (
            adjusted,
            f"Manual horizontal calculation endpoint selected at {selected_end:.3f} s. "
            f"Start remains fixed at {start:.3f} s. Inspect the green window, then save.",
            selected_end,
        )

    @app.callback(
        Output("sync-preview-result", "data", allow_duplicate=True),
        Output("sync-preview-status", "children", allow_duplicate=True),
        Input("sync-calculation-end", "value"),
        State("sync-preview-result", "data"),
        State("sync-index", "data"),
        prevent_initial_call=True,
    )
    def adjust_horizontal_calculation_end(selected_end, preview, index):
        """Use the endpoint slider to change only a horizontal end time."""
        if selected_end is None or preview is None:
            return no_update, no_update
        condition = conditions[min(max(int(index or 0), 0), len(conditions) - 1)]
        direction, _bws = _parse_condition(condition)
        if direction != "horizontal":
            return no_update, no_update
        try:
            selected_end = float(selected_end)
            start = float(preview["analysis_start_s"])
            current_end = float(preview["analysis_end_s"])
        except (KeyError, TypeError, ValueError):
            return no_update, "Synchronize the condition first, then select an endpoint."
        if np.isclose(selected_end, current_end):
            return no_update, no_update
        if selected_end <= start:
            return no_update, (
                f"The endpoint must be after the fixed start at {start:.3f} s. "
                "Move the endpoint slider farther to the right."
            )

        adjusted = dict(preview)
        adjusted["analysis_end_s"] = selected_end
        adjusted["analysis_window_method"] = (
            "horizontal: manual endpoint selected with endpoint slider"
        )
        adjusted["analysis_return_detected"] = True
        return (
            adjusted,
            f"Manual horizontal calculation endpoint selected at {selected_end:.3f} s. "
            f"Start remains fixed at {start:.3f} s. Inspect the green window, then save.",
        )

    @app.callback(
        Output("sync-index", "data"),
        Output("sync-status", "children"),
        Input("sync-run-next", "n_clicks"),
        Input("sync-previous", "n_clicks"),
        State("sync-index", "data"),
        State("sync-emg-channel", "value"),
        State("sync-force-channel", "value"),
        State("sync-emg-window", "value"),
        State("sync-force-window", "value"),
        State("sync-preview-result", "data"),
        prevent_initial_call=True,
    )
    def navigate_sync(_save_clicks, _previous_clicks, index, emg_channel,
                      force_channel, emg_window, force_window, preview):
        index = min(max(int(index or 0), 0), len(conditions) - 1)
        if ctx.triggered_id == "sync-previous":
            return max(0, index - 1), "Moved back. The saved synchronization result will be loaded."

        condition = conditions[index]
        saved_windows = _saved_window_selection(detection_report_path, condition)
        if preview is None:
            return index, "Detect peaks/dips and inspect the alignment before saving."
        if saved_windows is None or not _window_selection_matches(
                saved_windows, emg_channel, force_channel, emg_window, force_window):
            return index, "Save the current peak/dip windows, then detect the alignment again."
        if (preview.get("emg_channel") != emg_channel or
                preview.get("force_channel") != force_channel or
                not np.allclose([preview.get("emg_window_start_s"), preview.get("emg_window_end_s")], emg_window) or
                not np.allclose([preview.get("force_window_start_s"), preview.get("force_window_end_s")], force_window)):
            return index, "The windows or selected channels changed. Detect the alignment again before saving."
        if preview.get("analysis_start_s") is None or preview.get("analysis_end_s") is None:
            return index, "The calculation window could not be identified. Adjust the force alignment and try again."

        _save_report(report_path, preview)
        status = (
            f"Saved {_condition_label(condition)}.\n"
            f"EMG peaks ({emg_channel}): {np.round(preview['emg_peaks_s'], 3).tolist()} s\n"
            f"Position valleys ({force_channel}): {np.round(preview['force_peaks_s'], 3).tolist()} s\n"
            f"Saved force shift: {preview['offset_s']:+.3f} s\n"
            f"Calculation window: {preview['analysis_start_s']:.3f}–{preview['analysis_end_s']:.3f} s\n"
        )
        if index < len(conditions) - 1:
            return index + 1, status + "Continue with the next condition."
        return index, status + "All available synchronization results were saved. To finish, press Ctrl+C in the terminal."

    return app


# Compatibility name for callers that only want to open the crop stage.
create_app = create_crop_app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="data folder containing EMG/ and Force/")
    parser.add_argument("--participant", type=int, help="participant number")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    root = get_data_folder(args.data)
    emg_dir, force_dir = data_loading.find_data_folders(root)
    available = data_loading.list_participants(emg_dir, force_dir)
    if not available:
        raise SystemExit("No participant files found.")
    participant = args.participant if args.participant is not None else ask_participant(available)
    if participant not in available:
        raise SystemExit(f"Participant {participant} not found. Available: {available}")

    data = data_loading.load_all(emg_dir, force_dir, participant)
    emg_processing.process_all(data, calculate_outcomes=False)
    output_dir = output_subdirectory(root, "summaries", cfg.OUTPUT_FOLDER)
    crop_app = create_crop_app(data, participant, output_dir)
    print("Stage 1: save every crop, then stop the server with Ctrl+C to open Stage 2.")
    try:
        crop_app.run(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        pass

    from .cli import _load_saved_crops
    if not all_crops_saved(data, output_dir, participant):
        raise SystemExit("Stage 1 stopped before every condition was saved.")
    _loaded, missing = _load_saved_crops(data, output_dir, participant)
    if missing:
        raise SystemExit("Saved crop files could not be loaded for: " + ", ".join(missing))

    sync_app = create_sync_app(data, participant, output_dir)
    print("Stage 2: save every synchronization result, then stop the server to finish.")
    try:
        sync_app.run(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
