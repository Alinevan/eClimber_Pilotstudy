#!/usr/bin/env python3
"""Interactive manual force/EMG synchronisation.

The existing batch workflow uses the fixed ``SYNC_WINDOW_S`` window in
``synchronisation.py``. This tool is the alternative for trials where the
force recording is shifted: crop the derived signals, select a time window
directly on the raw force plot, detect either position valleys or signal peaks
in that window, and inspect the resulting shift before saving it to a report.

Usage:
    python manual_synchronisation.py
    python manual_synchronisation.py --data /path/to/data --participant 3

The browser view shows the processed EMG envelopes for the left and right
biceps and every numeric raw-force column. Cropping and saving only affects
derived files; the original recordings are not modified. Click the
box-select tool in the plot toolbar and drag across the force trace to select
the extrema-detection window. Then click "Detect peaks and synchronize".
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from plotly.subplots import make_subplots

import config as cfg
import data_loading
import emg_processing
import synchronisation
from main import ask_data_folder, ask_participant


def _condition_values(data):
    """Return condition keys as ``direction|bws`` strings for Dash dropdowns."""
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
    available = list(entry.get("env", {}))
    preferred = [name for name in ("L.Biceps Br.", "R.Biceps Br.") if name in available]
    fallback = [name for name in available if "biceps" in name.lower() and name not in preferred]
    return preferred + fallback


def _selected_x_range(selected_data):
    """Extract the horizontal range from Plotly box/lasso selection data."""
    if not selected_data:
        return None
    selected_range = selected_data.get("range", {}).get("x")
    if selected_range and len(selected_range) == 2:
        start, end = (float(selected_range[0]), float(selected_range[1]))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            return start, end
    xs = []
    for point in selected_data.get("points", []):
        x = point.get("x")
        if isinstance(x, (int, float, np.number)) and np.isfinite(x):
            xs.append(float(x))
    if len(xs) < 2:
        return None
    return min(xs), max(xs)


def _format_condition_options(values):
    return [{"label": _condition_label(value), "value": value} for value in values]


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
    """Save derived EMG envelopes and raw force values for one crop selection."""
    direction, bws = _parse_condition(condition)
    emg = data[direction]["emg"][bws]
    force = data[direction]["force"][bws]
    crop_dir = output_dir / "manual_crops" / f"{cfg.PARTICIPANT_PREFIX}{participant:02d}"
    crop_dir.mkdir(parents=True, exist_ok=True)

    emg_mask = _crop_mask(np.asarray(emg["t"], dtype=float), crop_windows["emg"])
    emg_out = pd.DataFrame({"time_s": np.asarray(emg["t"])[emg_mask]})
    for channel, signal in emg.get("env", {}).items():
        emg_out[channel] = np.asarray(signal)[emg_mask]
    emg_path = crop_dir / f"{direction}_BWS{bws}_filtered_emg.csv"
    emg_out.to_csv(emg_path, index=False)

    force_t = np.asarray(force["t"], dtype=float)
    force_mask = _crop_mask(force_t, crop_windows["force"])
    force_out = force["forces"].iloc[np.flatnonzero(force_mask)].copy().reset_index(drop=True)
    force_out.insert(0, "time_s", force_t[force_mask])
    force_path = crop_dir / f"{direction}_BWS{bws}_raw_force.csv"
    force_out.to_csv(force_path, index=False)
    return emg_path, force_path


def _make_figure(data, condition, force_channel=None, selected_window=None,
                 emg_window=None, sync_result=None, emg_crop=None,
                 force_crop=None, detection_mode="peaks"):
    """Build the filtered-EMG/raw-force figure for one condition."""
    direction, bws = _parse_condition(condition)
    emg = data[direction]["emg"][bws]
    force = data[direction]["force"][bws]
    biceps = _biceps_channels(emg)
    force_channels = list(force["forces"].columns)
    if force_channel not in force_channels:
        force_channel = force_channels[0] if force_channels else None

    titles = [f"Filtered EMG envelope: {channel}" for channel in biceps]
    titles += [f"Raw force: {channel}" for channel in force_channels]
    if not titles:
        titles = ["No channels available"]

    fig = make_subplots(rows=len(titles), cols=1, shared_xaxes=True,
                        subplot_titles=titles,
                        vertical_spacing=min(0.06, 0.6 / len(titles)))
    emg_t = np.asarray(emg["t"], dtype=float)
    emg_mask = _crop_mask(emg_t, emg_crop)
    force_t = np.asarray(force["t"], dtype=float)
    force_mask = _crop_mask(force_t, force_crop)

    row = 1
    for channel in biceps:
        fig.add_trace(
            go.Scatter(x=emg_t[emg_mask], y=np.asarray(emg["env"][channel])[emg_mask], mode="lines",
                       name=channel, legendgroup=f"emg-{channel}",
                       line=dict(width=1.2)),
            row=row, col=1,
        )
        row += 1

    force_offset = float(sync_result["offset_s"]) if sync_result else 0.0
    force_rows = {}
    for channel in force_channels:
        # The y-values remain raw. Only the x-axis is shifted after the user
        # applies a synchronization result.
        fig.add_trace(
            go.Scattergl(x=force_t[force_mask] + force_offset,
                         y=force["forces"][channel].to_numpy()[force_mask], mode="lines",
                         name=channel, legendgroup=f"force-{channel}",
                         line=dict(width=1.0)),
            row=row, col=1,
        )
        force_rows[channel] = row
        row += 1

    if selected_window:
        start, end = selected_window
        if sync_result:
            start += force_offset
            end += force_offset
        fig.add_vrect(x0=start, x1=end, fillcolor="royalblue", opacity=0.12,
                      line_width=1, line_dash="dash", row="all", col=1)
    if emg_window:
        fig.add_vrect(x0=emg_window[0], x1=emg_window[1], fillcolor="orange",
                      opacity=0.08, line_width=1, line_dash="dot", row="all", col=1)

    if sync_result:
        for peak in sync_result["emg_peaks_s"]:
            fig.add_vline(x=peak, line_color="darkorange", line_dash="dot",
                          line_width=1, row="all", col=1)
        for peak in sync_result["force_peaks_s"]:
            fig.add_vline(x=peak + force_offset, line_color="royalblue",
                          line_dash="dash", line_width=1, row="all", col=1)
        if force_channel in force_rows:
            plotted_t = force_t[force_mask] + force_offset
            plotted_y = force["forces"][force_channel].to_numpy()[force_mask]
            detected_t = np.asarray(sync_result["force_peaks_s"], dtype=float) + force_offset
            fig.add_trace(
                go.Scatter(x=detected_t, y=np.interp(detected_t, plotted_t, plotted_y),
                           mode="markers", name="Detected force extrema",
                           legendgroup="detected-extrema",
                           marker=dict(color="royalblue", size=9, symbol="x")),
                row=force_rows[force_channel], col=1,
            )

    for axis_row in range(1, len(titles) + 1):
        fig.update_xaxes(matches="x", row=axis_row, col=1)
    fig.update_xaxes(title_text="Time (s)", row=len(titles), col=1)
    fig.update_layout(
        title=(f"{_condition_label(condition)} — filtered EMG and raw force"
               + (f" — force shifted {force_offset:+.3f} s" if sync_result else "")
               + (" — position valleys" if detection_mode == "valleys" else " — signal peaks")
               + f"<br><sup>EMG: {emg['file']} &nbsp; | &nbsp; Force: {force['file']}</sup>"),
        template="plotly_white",
        height=max(700, 230 * len(titles)),
        margin=dict(t=90, r=40, b=60, l=70),
        hovermode="x unified",
        dragmode="select",
        legend=dict(groupclick="togglegroup"),
    )
    if biceps:
        fig.update_yaxes(title_text="EMG envelope", row=1, col=1)
    for force_row in range(len(biceps) + 1, len(titles) + 1):
        fig.update_yaxes(title_text="Raw force", row=force_row, col=1)
    return fig


def _initial_emg_window(data, condition, crop_window=None):
    direction, bws = _parse_condition(condition)
    duration = float(data[direction]["emg"][bws].get("duration", 0.0))
    start = float(crop_window[0]) if crop_window else 0.0
    end = float(crop_window[1]) if crop_window else duration
    return [start, min(end, start + cfg.SYNC_WINDOW_S)]


def _report_path(output_dir, participant):
    return output_dir / f"manual_sync_report_{cfg.PARTICIPANT_PREFIX}{participant:02d}.csv"


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


def create_app(data, participant, output_dir):
    conditions = _condition_values(data)
    if not conditions:
        raise SystemExit("No conditions contain both an EMG file and a force file.")
    default_condition = conditions[0]
    direction, bws = _parse_condition(default_condition)
    default_force_channels = list(data[direction]["force"][bws]["forces"].columns)
    default_detection_mode = "valleys" if any("position" in str(c).lower()
                                              for c in default_force_channels) else "peaks"
    default_force_channel = _preferred_force_channel(default_force_channels, default_detection_mode)
    default_crop_windows = _initial_crop_windows(data, default_condition)
    default_emg_window = _initial_emg_window(data, default_condition, default_crop_windows["emg"])

    app = Dash(__name__)
    app.title = "Manual EMG-force synchronisation"
    app.layout = html.Div([
        html.H2("Manual EMG–force synchronisation"),
        html.P([
            "The EMG traces are filtered envelopes. Force traces are raw. "
            "Use the crop controls first if the recording contains extra "
            "segments. Then use the box-select tool (square icon) and drag "
            "across the force trace to choose the peak/valley detection window. "
            "Plotly's legend "
            "toggles the left/right biceps and force channels."
        ]),
        html.Div([
            html.Label("Condition"),
            dcc.Dropdown(id="condition", options=_format_condition_options(conditions),
                         value=default_condition, clearable=False),
            html.Label("Synchronization signal"),
            dcc.Dropdown(
                id="detection-mode",
                options=[
                    {"label": "Position valleys (three most prominent dips)", "value": "valleys"},
                    {"label": "Signal peaks (legacy mode)", "value": "peaks"},
                ],
                value=default_detection_mode,
                clearable=False,
            ),
            html.Label("Force channel used for peak/valley detection"),
            dcc.Dropdown(id="force-channel",
                         options=[{"label": c, "value": c} for c in default_force_channels],
                         value=default_force_channel, clearable=False),
            html.Label("Filtered EMG crop window"),
            dcc.RangeSlider(
                id="emg-crop-window", min=0, max=max(default_crop_windows["emg"][1], 1.0),
                value=default_crop_windows["emg"], allowCross=False,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("Raw force crop window"),
            dcc.RangeSlider(
                id="force-crop-window", min=0, max=max(default_crop_windows["force"][1], 1.0),
                value=default_crop_windows["force"], allowCross=False,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("EMG peak-detection window (the orange band)"),
            dcc.RangeSlider(id="emg-window", min=0, max=max(default_emg_window[1], 1.0),
                            value=default_emg_window, allowCross=False,
                            tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"maxWidth": "900px", "display": "grid", "gap": "8px"}),
        html.Div([
            html.Button("Apply crop to view", id="apply-crop", n_clicks=0),
            html.Button("Save cropped derived files", id="save-crop", n_clicks=0),
        ], style={"display": "flex", "gap": "8px", "margin": "12px 0"}),
        html.Div(id="crop-status", style={"whiteSpace": "pre-wrap", "marginBottom": "12px"}),
        html.Div(id="selection-status", children="Select a window on the force plot."),
        html.Button("Detect peaks and synchronize", id="apply-sync", n_clicks=0,
                    style={"margin": "12px 0"}),
        html.Div(id="sync-status", style={"whiteSpace": "pre-wrap", "marginBottom": "12px"}),
        dcc.Graph(
            id="sync-graph",
            figure=_make_figure(data, default_condition, default_force_channel,
                                emg_window=default_emg_window,
                                emg_crop=default_crop_windows["emg"],
                                force_crop=default_crop_windows["force"],
                                detection_mode=default_detection_mode),
            config={"displaylogo": False, "modeBarButtonsToAdd": ["select2d", "lasso2d"]},
        ),
    ])

    @app.callback(
        Output("force-channel", "options"),
        Output("force-channel", "value"),
        Output("emg-window", "max"),
        Output("emg-window", "value"),
        Output("emg-crop-window", "max"),
        Output("emg-crop-window", "value"),
        Output("force-crop-window", "max"),
        Output("force-crop-window", "value"),
        Input("condition", "value"),
        Input("detection-mode", "value"),
        State("emg-crop-window", "value"),
        State("force-crop-window", "value"),
        State("emg-window", "value"),
    )
    def condition_changed(condition, detection_mode, current_emg_crop,
                          current_force_crop, current_emg_window):
        direction, bws = _parse_condition(condition)
        force_channels = list(data[direction]["force"][bws]["forces"].columns)
        preferred = _preferred_force_channel(force_channels, detection_mode)
        if ctx.triggered_id == "condition":
            crop_windows = _initial_crop_windows(data, condition)
            emg_window = _initial_emg_window(data, condition, crop_windows["emg"])
        else:
            crop_windows = {"emg": current_emg_crop, "force": current_force_crop}
            emg_window = current_emg_window
        emg_max = max(float(crop_windows["emg"][1]), 1.0)
        force_max = max(float(crop_windows["force"][1]), 1.0)
        return ([{"label": c, "value": c} for c in force_channels], preferred,
                emg_max, emg_window, emg_max, crop_windows["emg"],
                force_max, crop_windows["force"])

    @app.callback(
        Output("selection-status", "children"),
        Input("sync-graph", "selectedData"),
    )
    def selection_changed(selected_data):
        selected = _selected_x_range(selected_data)
        if selected is None:
            return "Select a window on the force plot with the box-select tool."
        return f"Selected force window: {selected[0]:.3f}–{selected[1]:.3f} s"

    @app.callback(
        Output("sync-graph", "figure"),
        Output("sync-status", "children"),
        Output("sync-graph", "selectedData"),
        Input("condition", "value"),
        Input("apply-sync", "n_clicks"),
        Input("emg-window", "value"),
        Input("apply-crop", "n_clicks"),
        Input("detection-mode", "value"),
        Input("emg-crop-window", "value"),
        Input("force-crop-window", "value"),
        State("force-channel", "value"),
        State("sync-graph", "selectedData"),
        prevent_initial_call=True,
    )
    def update_graph(condition, _n_clicks, emg_window, _crop_clicks, detection_mode,
                     emg_crop, force_crop, force_channel, selected_data):
        # Changing the condition, crop, mode, or EMG window only redraws the
        # view. The selected force window is used for detection only after the
        # synchronization button is clicked.
        emg_window = _clamp_window(emg_window, emg_crop)
        if ctx.triggered_id == "apply-crop":
            return _make_figure(data, condition, force_channel, emg_window=emg_window,
                                emg_crop=emg_crop, force_crop=force_crop,
                                detection_mode=detection_mode), \
                "Crop applied to the displayed derived signals. Original files are unchanged.", None
        if ctx.triggered_id != "apply-sync":
            return _make_figure(data, condition, force_channel, emg_window=emg_window,
                                emg_crop=emg_crop, force_crop=force_crop,
                                detection_mode=detection_mode), "", selected_data

        selected = _selected_x_range(selected_data)
        direction, bws = _parse_condition(condition)
        emg = data[direction]["emg"][bws]
        force = data[direction]["force"][bws]
        if selected is None:
            return _make_figure(data, condition, force_channel, emg_window=emg_window,
                                emg_crop=emg_crop, force_crop=force_crop,
                                detection_mode=detection_mode), \
                "No force window selected. Use the box-select tool on a force trace first.", selected_data
        if force_channel not in force["forces"]:
            return _make_figure(data, condition, force_channel, selected, emg_window,
                                emg_crop=emg_crop, force_crop=force_crop,
                                detection_mode=detection_mode), \
                f"Force channel not found: {force_channel}", selected_data

        emg_label = cfg.SYNC_EMG_LABEL if cfg.SYNC_EMG_LABEL in emg["env"] else (
            _biceps_channels(emg)[0] if _biceps_channels(emg) else None
        )
        if emg_label is None:
            return _make_figure(data, condition, force_channel, selected, emg_window,
                                emg_crop=emg_crop, force_crop=force_crop,
                                detection_mode=detection_mode), \
                "No EMG channel is available for synchronization.", selected_data
        emg_window = _clamp_window(emg_window, emg_crop)
        emg_peaks = synchronisation.find_sync_peaks_in_window(
            emg["t"], emg["env"][emg_label], emg_window[0], emg_window[1]
        )
        force_invert = detection_mode == "valleys"
        force_peaks = synchronisation.find_sync_peaks_in_window(
            force["t"], force["forces"][force_channel].to_numpy(),
            selected[0], selected[1], cfg.SYNC_FORCE_LP_HZ,
            force_invert if detection_mode == "valleys" else cfg.SYNC_FORCE_INVERT
        )
        if emg_peaks is None or force_peaks is None:
            missing = []
            if emg_peaks is None:
                missing.append(f"fewer than {cfg.SYNC_N_PEAKS} EMG peaks")
            if force_peaks is None:
                missing.append(f"fewer than {cfg.SYNC_N_PEAKS} force peaks")
            return _make_figure(data, condition, force_channel, selected, emg_window,
                                emg_crop=emg_crop, force_crop=force_crop,
                                detection_mode=detection_mode), \
                "Could not synchronize: " + "; ".join(missing) + ".", selected_data

        offset = float(np.mean(emg_peaks - force_peaks))
        spread = float(np.ptp(emg_peaks - force_peaks))
        result = {
            "direction": direction,
            "bws_percent": bws,
            "force_file": force["file"],
            "emg_file": emg["file"],
            "emg_channel": emg_label,
            "force_channel": force_channel,
            "force_window_start_s": selected[0],
            "force_window_end_s": selected[1],
            "emg_window_start_s": emg_window[0],
            "emg_window_end_s": emg_window[1],
            "emg_crop_start_s": emg_crop[0],
            "emg_crop_end_s": emg_crop[1],
            "force_crop_start_s": force_crop[0],
            "force_crop_end_s": force_crop[1],
            "detection_mode": detection_mode,
            "offset_s": offset,
            "peak_offset_spread_s": spread,
            "emg_peaks_s": np.round(emg_peaks, 6).tolist(),
            "force_peaks_s": np.round(force_peaks, 6).tolist(),
            "synchronised": True,
        }
        report = _report_path(output_dir, participant)
        _save_report(report, result)
        message = (
            f"Synchronized using {emg_label} and {force_channel} ({detection_mode}).\n"
            f"EMG peaks: {np.round(emg_peaks, 3).tolist()} s\n"
            f"Force extrema: {np.round(force_peaks, 3).tolist()} s\n"
            f"Applied force shift: {offset:+.3f} s\n"
            f"Peak-offset spread: {spread * 1000:.0f} ms\n"
            f"Saved: {report}"
        )
        return _make_figure(data, condition, force_channel, selected, emg_window, result,
                            emg_crop=emg_crop, force_crop=force_crop,
                            detection_mode=detection_mode), message, selected_data

    @app.callback(
        Output("crop-status", "children"),
        Input("save-crop", "n_clicks"),
        State("condition", "value"),
        State("emg-crop-window", "value"),
        State("force-crop-window", "value"),
        prevent_initial_call=True,
    )
    def save_crop(_n_clicks, condition, emg_crop, force_crop):
        emg_path, force_path = _save_cropped_files(
            data, condition, {"emg": emg_crop, "force": force_crop}, output_dir, participant
        )
        return (
            "Saved derived cropped files (raw source files unchanged):\n"
            f"{emg_path}\n{force_path}"
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="data folder containing EMG/ and Force/")
    parser.add_argument("--participant", type=int, help="participant number")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    root = args.data.expanduser().resolve() if args.data else ask_data_folder()
    emg_dir, force_dir = data_loading.find_data_folders(root)
    available = data_loading.list_participants(emg_dir, force_dir)
    if not available:
        raise SystemExit("No participant files found.")
    participant = args.participant if args.participant is not None else ask_participant(available)
    if participant not in available:
        raise SystemExit(f"Participant {participant} not found. Available: {available}")

    data = data_loading.load_all(emg_dir, force_dir, participant)
    emg_processing.process_all(data)
    output_dir = root.parent / cfg.OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(data, participant, output_dir)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
