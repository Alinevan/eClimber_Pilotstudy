#!/usr/bin/env python3
"""Climbing pilot analysis: runs the steps in order.

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="optional override for the configured data folder")
    ap.add_argument("--normalize", action="store_true", help="normalise envelopes to peak per muscle")
    ap.add_argument("--emg-unit", default=None, help="unit label for EMG axes")
    ap.add_argument("--show", action="store_true", help="also open the figures in the browser")
    ap.add_argument("--manual-sync-report", type=Path,
                    help="apply offsets from manual_sync_report_P##.csv instead of fixed-window detection")
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

    print("2. Processing EMG")
    emg_processing.process_all(data)

    print("3. Synchronising force to EMG")
    if args.manual_sync_report:
        sync = synchronisation.synchronize_from_report(data, args.manual_sync_report)
    else:
        # Legacy fixed-window path. It is intentionally kept as the batch
        # fallback; use manual_synchronisation.py when the force recording
        # starts later or the first fixed SYNC_WINDOW_S seconds do not contain
        # the sync peaks.
        sync = synchronisation.synchronize(data)
    outcalc.save_table(sync, summaries / f"sync_report_{tag}.csv")

    print("4. Plotting")
    for direction in cfg.DIRECTIONS:
        if direction not in data:
            print(f"No data for {direction} climbing.")
            continue
        fig = plotting.make_figure(direction, data[direction],
                                   normalize=args.normalize, emg_unit=args.emg_unit)
        html = plots / f"{direction}_climbing_{tag}.html"
        fig.write_html(html, include_plotlyjs="cdn")
        print(f"Saved {html}")
        if args.show:
            fig.show()

    print("5. Output calculations")
    outcalc.save_table(outcalc.rms_table(data), summaries / f"emg_rms_summary_{tag}.csv")


if __name__ == "__main__":
    main()
