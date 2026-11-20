"""Publication-style EMG-by-BWS figure from saved outcome summaries.

The figure uses the bilateral mean normalized envelope, which is the primary
activation measure used in the paper-style results.  It reads the saved CSV
summaries and therefore does not rerun signal processing, synchronization, or
climbing-window selection.

Example:
    python -m emg_force.paper_figure \
        --summaries /path/to/Data_Analysis/Output/summaries \
        --output figure4_emg_activation.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MUSCLE_ORDER = [
    "Biceps Br.",
    "Med. Triceps",
    "Flex.Carp.R",
    "Infraspin.",
]

DISPLAY_NAMES = {
    "Biceps Br.": "Biceps brachii",
    "Med. Triceps": "Triceps brachii (medial)",
    "Flex.Carp.R": "Flexor carpi ulnaris",
    "Infraspin.": "Supraspinatus",
}

PARTICIPANT_STYLES = {
    "P01": {"color": "#111111", "symbol": "square", "label": "P01"},
    "P02": {"color": "#8a8a8a", "symbol": "circle", "label": "P02"},
}


def _condition_style(participant: str, direction: str) -> dict:
    """Return the paper-style line and marker style for one condition."""
    participant_style = PARTICIPANT_STYLES[participant]
    vertical = direction == "vertical"
    return {
        "color": participant_style["color"],
        "dash": "solid" if vertical else "dash",
        "symbol": participant_style["symbol"],
        "fill": participant_style["color"] if vertical else "white",
        "label": f"{participant} {direction}",
    }


def _validate_outcomes(outcomes: pd.DataFrame) -> None:
    required = {
        "participant",
        "direction",
        "bws_percent",
        "muscle",
        "side",
        "mean_envelope_normalized",
    }
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(
            "EMG outcome summary is missing required columns: "
            + ", ".join(sorted(missing))
        )


def make_emg_bws_figure(outcomes: pd.DataFrame) -> go.Figure:
    """Create the four-panel EMG activation figure from summary rows."""
    _validate_outcomes(outcomes)
    rows = outcomes[outcomes["side"].astype(str).str.lower() == "bilateral"].copy()
    rows["participant"] = rows["participant"].astype(str).str.upper()
    rows["direction"] = rows["direction"].astype(str).str.lower()
    rows["bws_percent"] = pd.to_numeric(rows["bws_percent"], errors="coerce")
    rows["mean_envelope_normalized"] = pd.to_numeric(
        rows["mean_envelope_normalized"], errors="coerce"
    )
    rows = rows.dropna(subset=["bws_percent", "mean_envelope_normalized"])

    available_muscles = [m for m in MUSCLE_ORDER if m in set(rows["muscle"])]
    if len(available_muscles) != 4:
        missing = [m for m in MUSCLE_ORDER if m not in available_muscles]
        raise ValueError("The EMG summary does not contain all four muscles: " + ", ".join(missing))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[DISPLAY_NAMES[m] for m in available_muscles],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    y_max = max(35.0, float(rows["mean_envelope_normalized"].max()) + 2.0)
    y_tick = 5
    shown_legend = set()

    for panel_index, muscle in enumerate(available_muscles):
        panel_row = panel_index // 2 + 1
        panel_col = panel_index % 2 + 1
        muscle_rows = rows[rows["muscle"] == muscle]

        for participant in ("P01", "P02"):
            for direction in ("vertical", "horizontal"):
                subset = muscle_rows[
                    (muscle_rows["participant"] == participant)
                    & (muscle_rows["direction"] == direction)
                ].sort_values("bws_percent")
                if subset.empty:
                    continue
                style = _condition_style(participant, direction)
                show_legend = panel_index == 0 and style["label"] not in shown_legend
                if show_legend:
                    shown_legend.add(style["label"])
                fig.add_trace(
                    go.Scatter(
                        x=subset["bws_percent"],
                        y=subset["mean_envelope_normalized"],
                        mode="lines+markers",
                        name=style["label"],
                        legendgroup=style["label"],
                        showlegend=show_legend,
                        line={"color": style["color"], "dash": style["dash"], "width": 1.5},
                        marker={
                            "symbol": style["symbol"],
                            "size": 7,
                            "color": style["fill"],
                            "line": {"color": style["color"], "width": 1.4},
                        },
                        customdata=subset[["participant", "direction", "bws_percent"]],
                        hovertemplate=(
                            "%{customdata[0]} %{customdata[1]}<br>"
                            "BWS: %{customdata[2]:.0f}%<br>"
                            "Mean normalized envelope: %{y:.2f}%<extra></extra>"
                        ),
                    ),
                    row=panel_row,
                    col=panel_col,
                )

        fig.update_xaxes(
            range=[20, 80],
            tickmode="array",
            tickvals=[25, 50, 75],
            ticks="outside",
            showgrid=False,
            row=panel_row,
            col=panel_col,
        )
        fig.update_yaxes(
            range=[0, y_max],
            dtick=y_tick,
            ticks="outside",
            showgrid=False,
            row=panel_row,
            col=panel_col,
        )

    fig.update_xaxes(title_text="BWS (% of body weight)", row=2, col=1)
    fig.update_xaxes(title_text="BWS (% of body weight)", row=2, col=2)
    fig.update_yaxes(title_text="Mean EMG (% peak)", row=1, col=1)
    fig.update_yaxes(title_text="Mean EMG (% peak)", row=2, col=1)
    fig.update_layout(
        template="simple_white",
        width=900,
        height=680,
        margin={"l": 80, "r": 30, "t": 95, "b": 80},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#111111"},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.12,
            "yanchor": "top",
            "font": {"size": 12},
        },
        hoverlabel={"font": {"family": "Arial, sans-serif"}},
    )
    return fig


def load_emg_outcomes(summaries: Path, participants: tuple[str, ...]) -> pd.DataFrame:
    tables = []
    for participant in participants:
        path = summaries / f"emg_outcomes_{participant}.csv"
        if not path.exists():
            raise FileNotFoundError(f"EMG outcome summary not found: {path}")
        tables.append(pd.read_csv(path))
    if not tables:
        raise ValueError("No participant summaries were supplied.")
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries",
        type=Path,
        required=True,
        help="folder containing emg_outcomes_P01.csv, emg_outcomes_P02.csv, etc.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output HTML path for the paper-style figure",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        default=["P01", "P02"],
        help="participant labels to include (default: P01 P02)",
    )
    args = parser.parse_args()

    outcomes = load_emg_outcomes(args.summaries, tuple(args.participants))
    figure = make_emg_bws_figure(outcomes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Embed Plotly so the figure opens without requiring an internet connection.
    figure.write_html(args.output, include_plotlyjs=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
