#!/usr/bin/env python3
"""Extract all usability questionnaires in the study data folder.

Usage:
    python -m usability.cli
    python -m usability.cli --participant 2
"""

import argparse
from pathlib import Path

from common.paths import output_subdirectory, resolve_data_root
from common.participants import participant_tag

from .extraction import extract_questionnaire
from .export import write_outputs


def find_questionnaire_folder(data_root: Path) -> Path:
    """Find the questionnaire folder by its usability/questionnaire name."""
    candidates = []
    for directory in sorted(data_root.iterdir()):
        if not directory.is_dir():
            continue
        name = directory.name.lower()
        if "usability" in name and ("questionnaire" in name or "fragebogen" in name):
            candidates.append(directory)
    if not candidates:
        raise SystemExit(f"No usability questionnaire folder found in {data_root}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="optional override for the configured data folder")
    parser.add_argument("--questionnaire-dir", type=Path, help="override automatic questionnaire-folder discovery")
    parser.add_argument("--participant", type=int, help="extract only one participant")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data)
    print(f"Data folder: {data_root}")
    questionnaire_dir = args.questionnaire_dir.expanduser().resolve() if args.questionnaire_dir else find_questionnaire_folder(data_root)
    files = sorted(
        path for path in questionnaire_dir.glob("*.docx")
        if not path.name.startswith("~$")
    )
    if args.participant is not None:
        suffix = f"_P{args.participant:02d}.docx"
        files = [path for path in files if path.name.lower().endswith(suffix.lower())]
    if not files:
        raise SystemExit(f"No questionnaire Word documents found in {questionnaire_dir}")

    records = []
    warnings = []
    for path in files:
        print(f"Reading {path.name}")
        try:
            extracted, file_warnings = extract_questionnaire(path)
        except Exception as exc:  # report one bad document without losing the others
            warnings.append(f"{path.name}: extraction failed: {exc}")
            continue
        records.extend(extracted)
        warnings.extend(f"{path.name}: {warning}" for warning in file_warnings)

    output_dir = output_subdirectory(data_root, "usability_questionnaire")
    if args.participant is not None:
        output_dir = output_dir / participant_tag(args.participant)
        output_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths = write_outputs(records, warnings, output_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    for path in paths.values():
        print(f"Saved {path}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see {paths['warnings_csv']})")


if __name__ == "__main__":
    main()
