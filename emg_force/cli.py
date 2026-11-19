#!/usr/bin/env python3
"""Manual-first climbing pilot analysis workflow.

The main entry point loads the participant, asks whether a complete manual
crop/synchronisation report should be reused, and otherwise opens the manual
browser interface. It never falls back to automatic synchronization.

Usage:
    pip install ezc3d numpy scipy pandas plotly
    python -m emg_force.cli  (uses the configured data folder and asks for the participant number)

Folder layout:
    <parent>/
        <data folder>/   <- the path you enter
            EMG/ ...     (.c3d files ending in _P03 etc.)
            Force/ ...   (.csv files ending in _P03 etc.)
        output/          <- created automatically (same parent as the data folder)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.participants import participant_tag
from common.paths import output_subdirectory, resolve_data_root

from . import config as cfg
from . import emg_processing, io as data_loading, output_calculations as outcalc
from . import plotting, synchronisation


def get_data_folder(explicit=None):
    """Return the configured data folder without prompting on normal runs."""
    root = resolve_data_root(explicit)
    print(f"  Data folder: {root}")
    return root


def ask_participant(available):
    while True:
        print(f"Available participants: {', '.join(str(p) for p in available)}")
        try:
            nr = int(input(f"Participant number (number after '_{cfg.PARTICIPANT_PREFIX}'): ").strip())
        except ValueError:
            continue
        if nr in available:
            return nr
        print(f"  Participant {nr} not found.")


def _manual_report_is_reusable(path, data=None):
    """Return whether a saved manual report covers the available conditions."""
    required = {"direction", "bws_percent", "offset_s", "analysis_start_s", "analysis_end_s"}
    if not path.exists():
        return False
    try:
        report = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return False
    if not required.issubset(report.columns) or report.empty:
        return False
    if not report["analysis_start_s"].notna().all() or not report["analysis_end_s"].notna().all():
        return False
    if data is None:
        return True
    expected = {(direction, bws)
                for direction, d in data.items()
                for bws in set(d.get("emg", {})) & set(d.get("force", {}))}
    reported_bws = pd.to_numeric(report["bws_percent"], errors="coerce")
    reported = {(str(direction), int(bws))
                for direction, bws in zip(report["direction"], reported_bws)
                if pd.notna(bws)}
    return expected.issubset(reported)


def _ask_reuse_report(path):
    answer = input(
        f"Existing manual crop/synchronisation windows found at {path}. "
        "Reuse them? [y/N]: "
    ).strip().lower()
    return answer in {"y", "yes"}


def _run_manual_setup(data, participant, summaries, host, port):
    """Run the sequential crop stage and then the sequential sync stage."""
    from .manual_synchronisation import all_crops_saved, create_crop_app, create_sync_app

    print("\nStage 1 of 2: opening the condition-by-condition crop interface.")
    print("Save every available EMG/force condition, then stop the server with Ctrl+C to open Stage 2.")
    app = create_crop_app(data, participant, summaries)
    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\nStage 1 closed.")

    if not all_crops_saved(data, summaries, participant):
        raise SystemExit(
            "Cropping was stopped before every available EMG/force condition was saved. "
            "Re-run the workflow and finish Stage 1 before continuing."
        )

    loaded, missing = _load_saved_crops(data, summaries, participant)
    if missing:
        raise SystemExit("Saved crop files could not be loaded for: " + ", ".join(missing))
    print(f"Loaded {loaded} saved condition crops.")
    print("\nStage 2 of 2: opening the condition-by-condition synchronization interface.")
    print("Save every available synchronization result, then stop the server with Ctrl+C to continue the analysis.")
    app = create_sync_app(data, participant, summaries)
    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\nStage 2 closed.")


def _load_saved_crops(data, summaries, participant):
    """Load saved rectified-EMG/raw-force crops when they are available."""
    crop_dir = summaries / "manual_crops" / f"{cfg.PARTICIPANT_PREFIX}{participant:02d}"
    missing = []
    loaded = 0
    for direction, d in data.items():
        for bws, emg in d.get("emg", {}).items():
            if bws not in d.get("force", {}):
                continue
            force = d["force"][bws]
            emg_path = crop_dir / f"{direction}_BWS{bws}_rectified_emg.csv"
            force_path = crop_dir / f"{direction}_BWS{bws}_raw_force.csv"
            if not emg_path.exists() or not force_path.exists():
                missing.append(f"{direction} {bws}%")
                continue

            emg_df = pd.read_csv(emg_path)
            force_df = pd.read_csv(force_path)
            if "time_s" not in emg_df or "time_s" not in force_df:
                missing.append(f"{direction} {bws}% (missing time_s)")
                continue

            emg_t = emg_df["time_s"].to_numpy(dtype=float)
            force_t = force_df["time_s"].to_numpy(dtype=float)
            emg_channels = {
                column: emg_df[column].to_numpy(dtype=float)
                for column in emg_df.columns if column != "time_s"
            }
            if len(emg_t) < 2 or not emg_channels:
                missing.append(f"{direction} {bws}% (empty EMG crop)")
                continue
            fs = 1.0 / np.nanmedian(np.diff(emg_t))
            emg["fs"] = float(fs)
            emg["t"] = emg_t
            emg["time_origin_s"] = float(emg_t[0])
            emg["rectified"] = emg_channels
            # RMS is unchanged by rectification, and the final envelope is
            # calculated from the rectified saved crop.
            emg["filtered"] = emg_channels
            emg["duration"] = float(emg_t[-1] - emg_t[0] + 1.0 / fs)

            force["t"] = force_t
            force["forces"] = force_df.drop(columns=["time_s"])
            loaded += 1

    return loaded, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="optional override for the configured data folder")
    ap.add_argument("--normalize", action="store_true", help="normalise envelopes to peak per muscle")
    ap.add_argument("--emg-unit", default=None, help="unit label for EMG axes")
    ap.add_argument("--show", action="store_true", help="also open the figures in the browser")
    ap.add_argument("--manual-sync-report", type=Path,
                    help="reuse a saved manual crop/synchronisation report")
    ap.add_argument("--host", default="127.0.0.1", help="host for the manual interface")
    ap.add_argument("--port", type=int, default=8050, help="port for the manual interface")
    args = ap.parse_args()

    root = get_data_folder(args.data)
    emg_dir, force_dir = data_loading.find_data_folders(root)
    print(f"  EMG folder:   {emg_dir.name}\n  Force folder: {force_dir.name}")

    available = data_loading.list_participants(emg_dir, force_dir)
    if not available:
        raise SystemExit(f"No files ending with '_{cfg.PARTICIPANT_PREFIX}<number>' found.")
    participant = ask_participant(available)
    tag = participant_tag(participant)

    plots = output_subdirectory(root, "plots/final", cfg.OUTPUT_FOLDER)
    summaries = output_subdirectory(root, "summaries", cfg.OUTPUT_FOLDER)

    print(f"\n1. Loading data of {tag}")
    data = data_loading.load_all(emg_dir, force_dir, participant)
    if not data:
        raise SystemExit("No recognised condition files (horizontal/vertical + BWS %) for this participant.")

    avail = outcalc.availability_table(data)
    outcalc.save_table(avail, summaries / f"data_availability_{tag}.csv")
    for _, r in avail[avail["note"] != "complete"].iterrows():
        print(f"  ! {r['direction']} {r['bws_percent']}%: {r['note']}")

    report_path = args.manual_sync_report or (summaries / f"manual_sync_report_{tag}.csv")
    print(f"  Checking existing manual synchronization report: {report_path}")
    if args.manual_sync_report and not report_path.exists():
        raise SystemExit(f"Manual synchronisation report not found: {report_path}")

    if args.manual_sync_report:
        if not _manual_report_is_reusable(report_path, data):
            raise SystemExit(
                f"Manual synchronisation report is incomplete or lacks analysis windows: {report_path}"
            )
        reuse_report = True
    elif _manual_report_is_reusable(report_path, data):
        reuse_report = _ask_reuse_report(report_path)
    else:
        if report_path.exists():
            print(
                "  An older manual synchronization report exists, but it is incomplete "
                "for the available paired conditions; the workflow will resume it condition by condition."
            )
        reuse_report = False

    if reuse_report:
        loaded_crops, missing_crops = _load_saved_crops(data, summaries, participant)
        if missing_crops:
            print("  ! Saved synchronization windows found, but some saved crops are missing:")
            for condition in missing_crops:
                print(f"    - {condition}")
            print("  Preparing the missing/raw conditions from the original recordings.")
        if loaded_crops == 0:
            print("  No reusable cropped signal files found; preparing the original recordings.")
        # Reused crops already contain rectified EMG. Only prepare entries
        # which were not loaded from saved crop files.
        for direction, d in data.items():
            for bws, entry in d.get("emg", {}).items():
                if "filtered" not in entry or "rectified" not in entry:
                    emg_processing.process_all(
                        {direction: {"emg": {bws: entry}, "force": {bws: d["force"].get(bws)}
                                     if bws in d.get("force", {}) else {}}},
                        calculate_outcomes=False,
                    )
    else:
        print("2. Preparing EMG for the manual workflow")
        emg_processing.process_all(data, calculate_outcomes=False)

    if not reuse_report:
        _run_manual_setup(data, participant, summaries, args.host, args.port)
        if not _manual_report_is_reusable(report_path, data):
            raise SystemExit(
                f"The manual synchronisation report is incomplete at {report_path}. "
                "Save every available EMG/force condition in Stage 2 before continuing."
            )

    print("3. Applying manually selected crop and synchronisation windows")
    sync = synchronisation.synchronize_from_report(data, report_path)
    outcalc.save_table(sync, summaries / f"sync_report_{tag}.csv")

    print("4. Calculating outcomes inside the synchronized climbing window")
    issues = emg_processing.calculate_outcomes_from_sync_report(data, sync)
    for issue in issues:
        print(f"  ! No windowed EMG outcomes: {issue}")

    print("5. Preparing outcome summaries for plots")
    emg_outcomes = outcalc.emg_outcomes_table(data, participant)
    cci_outcomes = outcalc.cci_table(data, participant)
    try:
        force_outcomes, force_issues = outcalc.force_outcomes_table(
            data, sync, force_dir, participant
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for issue in force_issues:
        print(f"  ! No force outcome: {issue}")

    print("6. Plotting")
    for direction in cfg.DIRECTIONS:
        if direction not in data:
            print(f"No data for {direction} climbing.")
            continue
        fig = plotting.make_figure(direction, data[direction],
                                   normalize=args.normalize, emg_unit=args.emg_unit,
                                   emg_outcomes=emg_outcomes,
                                   cci_outcomes=cci_outcomes,
                                   force_outcomes=force_outcomes)
        html = plots / f"{direction}_climbing_{tag}.html"
        fig.write_html(html, include_plotlyjs="cdn")
        print(f"Saved {html}")
        if args.show:
            fig.show()

    print("7. Output calculations")
    outcalc.save_table(outcalc.rms_table(data), summaries / f"emg_rms_summary_{tag}.csv")
    outcalc.save_table(emg_outcomes, summaries / f"emg_outcomes_{tag}.csv")
    outcalc.save_table(cci_outcomes, summaries / f"cci_outcomes_{tag}.csv")
    outcalc.save_table(force_outcomes, summaries / f"force_outcomes_{tag}.csv")


if __name__ == "__main__":
    main()
