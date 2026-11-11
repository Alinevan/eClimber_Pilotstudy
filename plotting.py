"""Step 4: interactive Plotly figure per climbing direction."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cfg


def color_for(bws: int) -> str:
    if bws in cfg.COLORS:
        return cfg.COLORS[bws]
    return cfg.FALLBACK_COLORS[bws % len(cfg.FALLBACK_COLORS)]


def add_peak_markers(fig, row, bws, peak_t, t, y):
    """Mark detected sync peaks (toggle together with their condition)."""
    if len(t) == 0:
        return
    fig.add_trace(go.Scatter(x=peak_t, y=np.interp(peak_t, t, y), mode="markers",
                             marker=dict(symbol="x", size=10, color=color_for(bws), line=dict(width=2)),
                             name=f"BWS {bws}% sync peaks", legendgroup=f"bws{bws}", showlegend=False,
                             hovertemplate="sync peak %{x:.3f} s<extra></extra>"), row=row, col=1)


def make_figure(direction, d, normalize=False, emg_unit=None):
    emg, force = d["emg"], d["force"]
    if emg_unit is None:
        emg_unit = next((c["unit"] for c in emg.values() if c.get("unit")), "µV")
    conditions = sorted(set(emg) | set(force))

    muscles = sorted({m for c in emg.values() for m in c["env"]})
    force_cols = []
    for c in force.values():
        for col in c["forces"].columns:
            if col not in force_cols:
                force_cols.append(col)

    # Normalisation: peak envelope per muscle across all conditions of this direction
    peak = {}
    if normalize:
        for m in muscles:
            vals = [np.max(c["env"][m]) for c in emg.values() if m in c["env"]]
            peak[m] = max(vals) if vals and max(vals) > 0 else 1.0

    titles = [f"EMG envelope: {m}" for m in muscles] + [f"Force: {c}" for c in force_cols] \
        + ["EMG RMS per muscle", "Integrated EMG (area under envelope)"]
    n_rows = len(titles)
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles,
                        vertical_spacing=min(0.06, 0.6 / n_rows))

    shown = set()

    def legend_kw(bws):
        first = bws not in shown
        shown.add(bws)
        return dict(name=f"BWS {bws}%", legendgroup=f"bws{bws}", showlegend=first,
                    line=dict(color=color_for(bws), width=1.4))

    row = 1
    env_unit = "% of peak" if normalize else emg_unit
    for m in muscles:
        for bws in conditions:
            if bws in emg and m in emg[bws]["env"]:
                y = emg[bws]["env"][m]
                if normalize:
                    y = 100.0 * y / peak[m]
                fig.add_trace(go.Scatter(x=emg[bws]["t"], y=y, mode="lines", fill="tozeroy",
                                         opacity=0.6, **legend_kw(bws)), row=row, col=1)
                if m == cfg.SYNC_EMG_LABEL and emg[bws].get("sync_peaks") is not None:
                    add_peak_markers(fig, row, bws, emg[bws]["sync_peaks"], emg[bws]["t"], y)
        fig.update_yaxes(title_text=env_unit, row=row, col=1)
        row += 1

    for col in force_cols:
        for bws in conditions:
            if bws in force and col in force[bws]["forces"].columns:
                y = force[bws]["forces"][col].to_numpy()
                fig.add_trace(go.Scattergl(x=force[bws]["t"], y=y, mode="lines", **legend_kw(bws)),
                              row=row, col=1)
                if col == force[bws].get("sync_col"):
                    add_peak_markers(fig, row, bws, force[bws]["sync_peaks"], force[bws]["t"], y)
        fig.update_yaxes(title_text="Force", row=row, col=1)
        row += 1

    # RMS and iEMG bars (same legend groups, so they toggle with the curves)
    for key, ylab in (("rms", f"RMS ({emg_unit})"), ("iemg", f"iEMG ({emg_unit}·s)")):
        for bws in conditions:
            if bws not in emg:
                continue
            kw = legend_kw(bws)
            kw.pop("line")
            fig.add_trace(go.Bar(x=muscles, y=[emg[bws][key].get(m, np.nan) for m in muscles],
                                 marker_color=color_for(bws), **kw), row=row, col=1)
        fig.update_yaxes(title_text=ylab, row=row, col=1)
        row += 1

    # Link all time axes (zoom one, zoom all); label the last time axis
    n_time = n_rows - 2
    for r in range(1, n_time + 1):
        fig.update_xaxes(matches="x", row=r, col=1)
    if n_time >= 1:
        fig.update_xaxes(title_text="Time (s)", row=n_time, col=1)

    missing = [f"{k} {b}%" for k, src in (("EMG", emg), ("Force", force))
               for b in sorted(set(conditions) | set(cfg.EXPECTED_BWS)) if b not in src]

    # Synchronisation is successful only when both the EMG and force-side
    # sync peaks were found for the same BWS condition.
    sync_failed = []
    for bws in sorted(set(emg) & set(force)):
        emg_ok = emg[bws].get("sync_peaks") is not None
        force_ok = (force[bws].get("sync_peaks") is not None
                    and force[bws].get("sync_col") is not None)
        if not (emg_ok and force_ok):
            sync_failed.append(f"BWS {bws}%")

    status_lines = []
    if missing:
        status_lines.append(f"Missing: {', '.join(missing)}")
    if sync_failed:
        status_lines.append(f"Synchronization failed for: {', '.join(sync_failed)}")
    subtitle = "".join(f"<br><sup>{line}</sup>" for line in status_lines)

    fig.update_layout(
        # Keep the title and missing-condition message above the legend.
        title=dict(
            text=f"{direction.capitalize()} climbing: EMG and force by body weight support{subtitle}",
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
        ),
        height=max(600, 230 * n_rows),
        margin=dict(t=155, r=40, b=60, l=70),
        barmode="group",
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
