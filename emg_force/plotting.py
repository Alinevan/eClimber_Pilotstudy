"""Step 4: interactive Plotly figure per climbing direction."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as cfg
from .emg_processing import channel_side_and_muscle
from .utils import lowpass


def color_for(bws: int) -> str:
    if bws in cfg.COLORS:
        return cfg.COLORS[bws]
    return cfg.FALLBACK_COLORS[bws % len(cfg.FALLBACK_COLORS)]


def _crop_bounds(entry, signal_length):
    """Return the saved EMG crop in the entry's original time base."""
    fs = entry.get("fs")
    if fs is None or signal_length == 0:
        return None
    fs = float(fs)
    origin = float(entry.get("time_origin_s", 0.0))
    recording_end = origin + (signal_length - 1) / fs
    start = float(entry.get("crop_start_s", origin))
    end = float(entry.get("crop_end_s", recording_end))
    start = min(max(start, origin), recording_end)
    end = min(max(end, start), recording_end)
    return start, end


def _full_emg_series(entry, channel):
    """Return the complete saved EMG crop as a plot-ready envelope.

    The final metrics remain restricted to the calculation window. This is a
    separate plotting series so the HTML figure can show the complete crop.
    """
    filtered = entry.get("filtered", {}).get(channel)
    if filtered is None:
        filtered = entry.get("rectified", {}).get(channel)
    fs = entry.get("fs")
    if filtered is None or fs is None:
        return None, None, None
    filtered = np.asarray(filtered, dtype=float)
    if len(filtered) < 2:
        return None, None, None
    fs = float(fs)
    origin = float(entry.get("time_origin_s", 0.0))
    bounds = _crop_bounds(entry, len(filtered))
    if bounds is None:
        return None, None, None
    start, end = bounds
    start_idx = max(0, int(round((start - origin) * fs)))
    end_idx = min(len(filtered), int(round((end - origin) * fs)) + 1)
    if end_idx <= start_idx:
        return None, None, None

    # Pad the selected crop for the low-pass filter, but return only the crop
    # itself so the plot starts and ends exactly at the saved crop limits.
    pad = max(1, int(round(fs)))
    padded_start = max(0, start_idx - pad)
    padded_end = min(len(filtered), end_idx + pad)
    try:
        envelope_padded = lowpass(
            np.abs(filtered[padded_start:padded_end]), cfg.ENVELOPE_LP_HZ, fs
        )
    except (ValueError, RuntimeError):
        envelope_padded = np.abs(filtered[padded_start:padded_end])
    envelope = envelope_padded[start_idx - padded_start:end_idx - padded_start]
    step = max(1, int(round(fs / cfg.PLOT_FS_HZ)))
    t = origin + np.arange(start_idx, end_idx, dtype=float) / fs
    return t[::step], envelope[::step], start


def _relative_calculation_window(entry):
    """Return the saved calculation window relative to the EMG crop start."""
    start = entry.get("analysis_start_s")
    end = entry.get("analysis_end_s")
    if start is None or end is None:
        return None
    try:
        start = float(start)
        end = float(end)
        crop_start = float(entry.get("crop_start_s", 0.0))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None
    return start - crop_start, end - crop_start


def add_peak_markers(fig, row, bws, peak_t, t, y, label,
                     marker_t=None, marker_y=None):
    """Mark all detected sync events as part of the condition legend group."""
    if peak_t is None or len(t) == 0:
        return
    peak_t = np.asarray(peak_t, dtype=float)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    if not finite.any():
        return
    finite_t = t[finite]
    finite_y = y[finite]
    if marker_t is not None and marker_y is not None:
        marker_t = np.asarray(marker_t, dtype=float)
        marker_y = np.asarray(marker_y, dtype=float)
        marker_finite = np.isfinite(marker_t) & np.isfinite(marker_y)
        marker_y = np.interp(peak_t, marker_t[marker_finite], marker_y[marker_finite])
    else:
        # Only interpolate inside the visible series. np.interp otherwise
        # silently uses the first/last value, which put sync crosses at zero
        # when sync peaks occurred before the calculation window.
        marker_y = np.full(len(peak_t), np.nan, dtype=float)
        inside = (peak_t >= finite_t[0]) & (peak_t <= finite_t[-1])
        marker_y[inside] = np.interp(peak_t[inside], finite_t, finite_y)
    # Synchronization pulses occur before the climbing calculation window, so
    # the envelope is NaN at those x-values. Put those markers at the top of
    # the visible axis instead of silently dropping them.
    marker_y = np.where(np.isfinite(marker_y), marker_y, np.nanmax(finite_y))
    fig.add_trace(go.Scatter(x=peak_t, y=marker_y, mode="markers",
                             marker=dict(symbol="x", size=10,
                                         color=cfg.SYNC_EMG_PEAK_COLOR),
                             name=f"BWS {bws}% {label}", legendgroup=f"bws{bws}",
                             showlegend=False, hovertemplate="%{x:.3f} s<extra></extra>"),
                  row=row, col=1)


def _display_muscle_name(muscle):
    """Return the configured display name for a side-independent muscle key."""
    for side in ("L", "R"):
        label = cfg.EMG_DISPLAY_LABELS.get(f"{side}.{muscle}")
        if label:
            return label[3:] if label.startswith(f"{side}. ") else label
    return muscle


def make_figure(direction, d, normalize=False, emg_unit=None,
                emg_outcomes=None, cci_outcomes=None, force_outcomes=None):
    emg, force = d["emg"], d["force"]
    if emg_unit is None:
        emg_unit = next((c["unit"] for c in emg.values() if c.get("unit")), "µV")
    conditions = sorted(set(emg) | set(force))

    available_muscles = {m for c in emg.values()
                         for m in c.get("filtered", c.get("env", {}))}
    # Keep left/right recordings adjacent so the same muscle can be compared
    # visually before moving on to the next muscle group.
    muscles = [m for m in cfg.EMG_LABELS if m in available_muscles]
    muscles.extend(sorted(available_muscles - set(muscles)))
    force_cols = []
    for c in force.values():
        for col in c["forces"].columns:
            if col not in force_cols:
                force_cols.append(col)

    # Normalisation: peak envelope per muscle across all conditions of this direction
    peak = {}
    if normalize:
        for m in muscles:
            vals = [np.nanmax(c["env"][m]) for c in emg.values()
                    if m in c["env"] and np.isfinite(c["env"][m]).any()]
            peak[m] = max(vals) if vals and max(vals) > 0 else 1.0

    titles = [f"EMG envelope: {cfg.EMG_DISPLAY_LABELS.get(m, m)}" for m in muscles] \
        + [f"Force: {c}" for c in force_cols]

    # The summary panels use bilateral values to match the paper's primary
    # outcomes. Left/right values remain available in the CSV summaries.
    bar_specs = []
    if emg_outcomes is not None and not emg_outcomes.empty:
        bilateral = emg_outcomes[
            (emg_outcomes["direction"].astype(str) == str(direction)) &
            (emg_outcomes["side"] == "bilateral")
        ]
        outcome_muscles = list(dict.fromkeys(bilateral["muscle"].tolist()))
        preferred = []
        for channel in cfg.EMG_LABELS:
            _side, muscle = channel_side_and_muscle(channel)
            if muscle in outcome_muscles and muscle not in preferred:
                preferred.append(muscle)
        outcome_muscles = preferred + [m for m in outcome_muscles if m not in preferred]
        if outcome_muscles:
            bar_specs.extend([
                ("emg", "mean_envelope_raw", "Mean raw envelope (bilateral)",
                 f"EMG amplitude ({emg_unit})", outcome_muscles),
                ("emg", "mean_envelope_normalized", "Mean normalized envelope (bilateral)",
                 "% of participant/muscle envelope peak", outcome_muscles),
                ("emg", "rms_raw", "Raw RMS (bilateral)",
                 f"RMS ({emg_unit})", outcome_muscles),
                ("emg", "rms_normalized", "Normalized RMS (bilateral)",
                 "% of participant/muscle RMS reference", outcome_muscles),
                ("emg", "iemg_raw", "Raw integrated EMG (bilateral)",
                 f"iEMG ({emg_unit}·s)", outcome_muscles),
                ("emg", "iemg_normalized", "Normalized integrated EMG (bilateral)",
                 "% peak·s", outcome_muscles),
            ])

    if cci_outcomes is not None and not cci_outcomes.empty:
        cci_rows = cci_outcomes[cci_outcomes["direction"].astype(str) == str(direction)]
        if not cci_rows.empty:
            bar_specs.append((
                "cci", None, "Biceps–triceps co-contraction index",
                "CCI (%)", ["Left CCI", "Right CCI", "Bilateral CCI"],
            ))

    if force_outcomes is not None and not force_outcomes.empty:
        force_rows = force_outcomes[
            force_outcomes["direction"].astype(str) == str(direction)
        ]
        if not force_rows.empty:
            bar_specs.extend([
                ("force", "force_N", "Force summary during climbing", "Force (N)",
                 ["Mean force", "Nominal force", "Peak force", "SD", "RMSE"]),
                ("force", "force_ratio", "Actual-to-nominal force ratio", "Ratio (-)",
                 ["Actual / nominal"]),
                ("force", "force_variability", "Force variability relative to target", "%",
                 ["CV", "SD / nominal"]),
            ])

    titles += [spec[2] for spec in bar_specs]
    n_rows = len(titles)
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles,
                        vertical_spacing=min(0.06, 0.6 / n_rows))

    shown = set()
    row_y_bounds = {}

    def record_y_bounds(row_number, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        current = row_y_bounds.get(row_number)
        bounds = (float(np.min(values)), float(np.max(values)))
        row_y_bounds[row_number] = bounds if current is None else (
            min(current[0], bounds[0]), max(current[1], bounds[1])
        )

    def legend_kw(bws):
        first = bws not in shown
        shown.add(bws)
        return dict(name=f"BWS {bws}%", legendgroup=f"bws{bws}", showlegend=first,
                    line=dict(color=color_for(bws), width=1.4))

    row = 1
    env_unit = "% of peak" if normalize else emg_unit
    for m in muscles:
        for bws in conditions:
            if bws in emg:
                t, y, crop_start = _full_emg_series(emg[bws], m)
                if t is None:
                    continue
                t = t - crop_start
                if normalize:
                    y = 100.0 * y / peak[m]
                fig.add_trace(go.Scatter(x=t, y=y, mode="lines", fill="tozeroy",
                                         opacity=0.6, **legend_kw(bws)), row=row, col=1)
                record_y_bounds(row, y)
                sync_channel = emg[bws].get("sync_channel", emg[bws].get("emg_channel", cfg.SYNC_EMG_LABEL))
                if m == sync_channel and emg[bws].get("sync_peaks") is not None:
                    peaks = np.asarray(emg[bws]["sync_peaks"], dtype=float) - crop_start
                    add_peak_markers(fig, row, bws, peaks, t, y, "EMG sync peak")
        fig.update_yaxes(title_text=env_unit, row=row, col=1)
        row += 1

    for col in force_cols:
        for bws in conditions:
            if bws in force and col in force[bws]["forces"].columns:
                y = force[bws]["forces"][col].to_numpy()
                # Force time has already received the synchronization offset
                # in synchronize_from_report. Rebase it to the EMG crop start
                # so the aligned EMG and force traces share the same zero.
                emg_origin = (emg.get(bws, {}).get("crop_start_s")
                              if bws in emg else None)
                if emg_origin is None:
                    emg_origin = float(force[bws].get("crop_start_s", 0.0))
                force_t = np.asarray(force[bws]["t"], dtype=float) - float(emg_origin)
                fig.add_trace(go.Scattergl(x=force_t, y=y, mode="lines", **legend_kw(bws)),
                              row=row, col=1)
                record_y_bounds(row, y)
                marker_t = force_t
                if col == force[bws].get("sync_col"):
                    peaks = force[bws].get("sync_peaks")
                    peaks = None if peaks is None else np.asarray(peaks, dtype=float) - float(emg_origin)
                    add_peak_markers(fig, row, bws, peaks, marker_t, y,
                                     "force sync peak")
                if col == force[bws].get("position_col") and force[bws].get("position_sync_peaks") is not None:
                    # Avoid drawing the same three crosses twice when the
                    # selected force channel is also the position channel.
                    if col != force[bws].get("sync_col"):
                        dips = np.asarray(force[bws]["position_sync_peaks"], dtype=float) - float(emg_origin)
                        add_peak_markers(fig, row, bws, dips, marker_t, y,
                                         "position sync valley")
        fig.update_yaxes(title_text="Force", row=row, col=1)
        row += 1

    # Summary bars use the same BWS legend groups as the time-series traces.
    for kind, metric, title, ylab, categories in bar_specs:
        if kind == "emg":
            rows = emg_outcomes[
                (emg_outcomes["direction"].astype(str) == str(direction)) &
                (emg_outcomes["side"] == "bilateral")
            ]
            x_values = [_display_muscle_name(muscle) for muscle in categories]
            for bws in conditions:
                subset = rows[rows["bws_percent"] == bws].set_index("muscle")
                if subset.empty:
                    continue
                kw = legend_kw(bws)
                kw.pop("line")
                fig.add_trace(go.Bar(
                    x=x_values,
                    y=[subset[metric].get(muscle, np.nan) for muscle in categories],
                    marker_color=color_for(bws), **kw,
                ), row=row, col=1)
        elif kind == "cci":
            rows = cci_outcomes[
                cci_outcomes["direction"].astype(str) == str(direction)
            ]
            x_values = categories
            for bws in conditions:
                subset = rows[rows["bws_percent"] == bws]
                if subset.empty:
                    continue
                values = subset.iloc[-1]
                kw = legend_kw(bws)
                kw.pop("line")
                fig.add_trace(go.Bar(
                    x=x_values,
                    y=[values.get("cci_left_percent", np.nan),
                       values.get("cci_right_percent", np.nan),
                       values.get("cci_bilateral_percent", np.nan)],
                    marker_color=color_for(bws), **kw,
                ), row=row, col=1)
        else:
            rows = force_outcomes[
                force_outcomes["direction"].astype(str) == str(direction)
            ]
            x_values = categories
            for bws in conditions:
                subset = rows[rows["bws_percent"] == bws]
                if subset.empty:
                    continue
                values = subset.iloc[-1]
                if metric == "force_N":
                    y_values = [values.get("mean_force_N", np.nan),
                                values.get("nominal_force_N", np.nan),
                                values.get("peak_force_N", np.nan),
                                values.get("force_SD_N", np.nan),
                                values.get("force_RMSE_N", np.nan)]
                elif metric == "force_ratio":
                    y_values = [values.get("actual_to_nominal_ratio", np.nan)]
                else:
                    y_values = [values.get("force_CV_percent", np.nan),
                                values.get("force_SD_percent_nominal", np.nan)]
                kw = legend_kw(bws)
                kw.pop("line")
                fig.add_trace(go.Bar(
                    x=x_values, y=y_values, marker_color=color_for(bws), **kw,
                ), row=row, col=1)
        fig.update_yaxes(title_text=ylab, row=row, col=1)
        row += 1

    # Link all time axes (zoom one, zoom all); label the last time axis
    n_time = len(muscles) + len(force_cols)
    for r in range(1, n_time + 1):
        fig.update_xaxes(matches="x", row=r, col=1)
        for bws in conditions:
            entry = emg.get(bws, {})
            calculation_window = _relative_calculation_window(entry)
            if calculation_window is not None:
                x0, x1 = calculation_window
                y0, y1 = row_y_bounds.get(r, (0.0, 1.0))
                if y0 == y1:
                    padding = max(abs(y0) * 0.05, 1.0)
                    y0 -= padding
                    y1 += padding
                # Use a normal Plotly trace instead of a layout shape. This
                # makes the shaded window belong to the BWS legend group, so
                # clicking a condition hides its signal, markers, and shade.
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, x1, x0], y=[y0, y0, y1, y1],
                        mode="none", fill="toself",
                        fillcolor=color_for(bws), opacity=0.10,
                        hoverinfo="skip",
                        name=f"BWS {bws}% calculation window",
                        legendgroup=f"bws{bws}", showlegend=False,
                    ),
                    row=r, col=1,
                )
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
