# eClimber Pilot Study Analysis

Python code for my Master's thesis: Assistive Load and Muscle Demand During Supported Climbing.

## Repository structure

The repository contains two independent analysis areas:

```text
common/       shared path and participant-ID helpers
emg_force/    EMG and force loading, processing, synchronisation, and plots
usability/    Word questionnaire extraction and overview exports
tests/        small automated checks for shared and questionnaire logic
```

There is intentionally no `pyproject.toml`. Install the packages listed in
`requirements.txt` in the active Python environment.

## EMG and force analysis

The EMG/force code:

1. Loads surface EMG data (`.c3d`) and eClimber force data (`.csv`);
2. pre-processes the EMG (filtering, rectification, smoothing);
3. aligns EMG and force data using synchronization peaks;
4. plots EMG and force for each condition; and
5. calculates summary outcomes.

Run the main workflow with:

```text
python -m emg_force.cli
```

For trials where the force recording is shifted, use the manual tool:

```text
python -m emg_force.manual_synchronisation --data /path/to/data --participant 3
```

To apply a saved manual synchronization report in the regular workflow:

```text
python -m emg_force.cli --manual-sync-report /path/to/output/manual_sync_report_P03.csv
```

The raw-data viewer can be run with:

```text
python -m emg_force.raw_data_viewer
```

## Study conditions

- Modality: vertical (top-rope) and horizontal (lateral bouldering)
- BWS levels: 25%, 50%, 75% of body weight (if feasible with patient)
- EMG: biceps brachii, triceps, infraspinatus, flexor ulnaris, erector spinae
- No MVC normalization

## Usability questionnaires

Questionnaire documents are read from a folder whose name contains
`Usability` and `questionnaire`, for example `Usability_questionnaire`.
The extractor reads Word tables, legacy checkbox fields, text fields, and
visible remarks. It ignores the U1–U4 research-question box.

Section E is reported with `Bemerkungen`, consistently with the other
questionnaire sections. The final `Generelle Bemerkungen` field is kept
separately.

The main data folder is configured once in `common/paths.py`, so it is used
automatically:

```text
python -m usability.cli
```

For one participant only:

```text
python -m usability.cli --participant 2
```

Participant-specific outputs are saved under `output/usability_questionnaire/P02/`.
Run the command without `--participant` to rebuild the aggregate workbook with
all questionnaire documents.

The optional `--data /path/to/Data` argument can override the configured path
when running the repository on another computer.

Questionnaire outputs are written separately to `output/usability_questionnaire/`:

The overview and Excel workbook always contain rows for `A_Bemerkungen` through
`F_Bemerkungen`, followed by `Generelle Bemerkungen`, even when a participant
left one of those fields empty.

- `usability_questionnaire_overview.csv`: one row per question/condition and one column per participant;
- `usability_questionnaire_long.csv`: one answer per row, suitable for further analysis;
- `usability_questionnaire_readout.txt`: readable participant-by-participant summary;
- `usability_questionnaire_warnings.csv`: missing or unusual fields; and
- `usability_questionnaire_overview.xlsx`: Excel workbook, regenerated from all questionnaire documents on every run. `openpyxl` is used for formatting when available; a basic workbook is also created without it.

All generated output is organized into two folders:

```text
output/
├── plots/
│   ├── raw/     raw-data HTML figures
│   └── final/   processed EMG/force HTML figures
├── summaries/   EMG/force CSV, text, and synchronization outputs
└── usability_questionnaire/  questionnaire CSV, Excel, and text outputs
```

## Tests

Tests do not run automatically during analysis. Run them when checking the
refactored code or after changing the questionnaire reader:

```text
python -m unittest discover -s tests
```

## Data

Data files are not part of this repository because they contain participant
data. Pass the data path on the command line when running an analysis.

## Author

Aline Vandebroek, 2026
