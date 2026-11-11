#!/usr/bin/env python3
"""Plot raw EMG and force recordings without analysis preprocessing.

Only file loading and condition identification are performed. This script
does not apply EMG filtering, rectification, envelope extraction, decimation,
force filtering, synchronization, cropping, or normalization.

Usage:
    python raw_data_viewer.py
    python raw_data_viewer.py --show

Expected folder layout:
    <data folder>/
        EMG/    (*.c3d)
        Force/  (*.csv)
"""
import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cfg
import data_loading
from main import ask_data_folder, ask_participant


def color_for(bws):
    return cfg.COLORS.get(bws, cfg.FALLBACK_COLORS[bws % len(cfg.FALLBACK_COLORS)])


def load_raw_trials(emg_dir, force_dir, participant):
    """Use the existing loader, while explicitly disabling its optional force filter."""
    original_force_filter = cfg.FORCE_LP_HZ
    cfg.FORCE_LP_HZ = None
    try:
        return data_loading.load_all(emg_dir, force_dir, participant)
    finally:
        cfg.FORCE_LP_HZ = original_force_filter


def make_raw_figure(direction, data):
    """Create one interactive raw-data figure for a climbing direction."""
    emg, force = data["emg"], data["force"]
    conditions = sorted(set(emg) | set(force))
    emg_channels = sorted({channel for entry in emg.values() for channel in entry["raw"]})
    force_channels = []
    for entry in force.values():
        for channel in entry["forces"].columns:
            if channel not in force_channels:
                force_channels.append(channel)

    titles = [f"Raw EMG: {channel}" for channel in emg_channels]
    titles += [f"Raw force: {channel}" for channel in force_channels]
    if not titles:
        raise ValueError(f"No EMG or force channels found for {direction}.")

    fig = make_subplots(
        rows=len(titles),
        cols=1,
        subplot_titles=titles,
        vertical_spacing=min(0.06, 0.6 / len(titles)),
    )

    shown = set()

    def trace_kwargs(bws):
        first = bws not in shown
        shown.add(bws)
        return dict(
            name=f"BWS {bws}%",
            legendgroup=f"bws{bws}",
            showlegend=first,
            line=dict(color=color_for(bws), width=1),
        )

    row = 1
    for channel in emg_channels:
        for bws in conditions:
            if bws not in emg or channel not in emg[bws]["raw"]:
                continue
            y = emg[bws]["raw"][channel]
            t = np.arange(len(y), dtype=float) / emg[bws]["fs"]
            fig.add_trace(
                go.Scattergl(x=t, y=y, mode="lines", **trace_kwargs(bws)),
                row=row,
                col=1,
            )
        row += 1

    for channel in force_channels:
        for bws in conditions:
            if bws not in force or channel not in force[bws]["forces"].columns:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=force[bws]["t"],
                    y=force[bws]["forces"][channel].to_numpy(),
                    mode="lines",
                    **trace_kwargs(bws),
                ),
                row=row,
                col=1,
            )
        row += 1

    for axis_row in range(1, len(titles) + 1):
        fig.update_xaxes(matches="x", row=axis_row, col=1)
    fig.update_xaxes(title_text="Raw acquisition time (s)", row=len(titles), col=1)

    missing = [
        f"{kind} {bws}%"
        for kind, source in (("EMG", emg), ("Force", force))
        for bws in sorted(set(cfg.EXPECTED_BWS) | set(conditions))
        if bws not in source
    ]
    missing_text = f"<br><sup>Missing: {', '.join(missing)}</sup>" if missing else ""
    fig.update_layout(
        title=dict(
            text=(
                f"{direction.capitalize()} climbing: raw EMG and force"
                f"<br><sup>No filtering, rectification, envelope extraction, decimation, "
                f"synchronization, cropping, or normalization applied.</sup>{missing_text}"
            ),
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
        ),
        height=max(700, 230 * len(titles)),
        margin=dict(t=125, r=40, b=60, l=70),
        template="plotly_white",
        legend=dict(
            title=dict(text="Condition (click to toggle)", side="left"),
            groupclick="togglegroup",
            orientation="h",
            yanchor="top",
            y=1.04,
            x=0,
            xanchor="left",
        ),
        hovermode="x unified",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="also open the figures in a browser")
    args = parser.parse_args()

    root = ask_data_folder()
    emg_dir, force_dir = data_loading.find_data_folders(root)
    print(f"  EMG folder:   {emg_dir.name}\n  Force folder: {force_dir.name}")

    available = data_loading.list_participants(emg_dir, force_dir)
    if not available:
        raise SystemExit(f"No files ending with '_{cfg.PARTICIPANT_PREFIX}<number>' found.")
    participant = ask_participant(available)

    data = load_raw_trials(emg_dir, force_dir, participant)
    if not data:
        raise SystemExit("No recognised raw condition files found.")

    output = root.parent / cfg.OUTPUT_FOLDER
    output.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.PARTICIPANT_PREFIX}{participant:02d}"
    for direction in cfg.DIRECTIONS:
        if direction not in data:
            continue
        figure = make_raw_figure(direction, data[direction])
        path = output / f"raw_{direction}_{tag}.html"
        figure.write_html(path, include_plotlyjs="cdn")
        print(f"Saved {path}")
        if args.show:
            figure.show()


if __name__ == "__main__":
    main()
