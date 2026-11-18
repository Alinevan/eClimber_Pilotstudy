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

### EMG/force script overview

| Script/module | Role |
|---|---|
| `emg_force/cli.py` | Main manual-first workflow and final output generation |
| `emg_force/manual_synchronisation.py` | Interactive EMG/force cropping, window selection, peak/dip detection, and report creation |
| `emg_force/io.py` | Loads and pairs EMG/force files by participant, direction, and BWS |
| `emg_force/emg_processing.py` | EMG filtering, rectification, windowed envelope, RMS, and iEMG calculations |
| `emg_force/synchronisation.py` | Detects events inside user-selected windows and applies saved manual synchronization reports |
| `emg_force/plotting.py` | Final synchronized HTML plots and event/window markers |
| `emg_force/output_calculations.py` | Availability and EMG summary CSV tables |
| `emg_force/raw_data_viewer.py` | Separate raw-data inspection tool; not part of the main workflow |
| `emg_force/config.py` | Central analysis settings and channel labels |

The main workflow is manual by design. The old fixed-window automatic
synchronization function has been removed from the active code.

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

The main workflow always uses the interactive manual crop and synchronisation
process. If a saved manual report already exists, the workflow asks whether it
should be reused. Otherwise it opens the manual interface:

```text
python -m emg_force.cli
```

The manual workflow is divided into two separate browser stages. Stage 1 walks
through every available paired direction/BWS condition in order. The user adjusts the
independently zoomable EMG and force/position plots, then presses “Save this
condition and go to next”; pressing it with the full range selected saves the
full recording. Previously saved ranges are reloaded when returning to a
condition. Conditions for which one raw file is absent are reported in the
availability table but are not included in the workflow count. After Stage 1 is
complete, Stage 2 loads only those saved crop files and walks through the same
available conditions for EMG peak-window and force-position dip-window
selection. Each synchronization is saved with “Save synchronization and go to
next”.

After synchronization, the green shaded calculation window is shown in both
the EMG and force/position plots. It starts one second after the final
synchronization valley. For vertical trials it ends at the post-synchronization
global position maximum; for horizontal trials it ends at a sustained return
toward the baseline around the final valley. For horizontal trials, click the
desired endpoint on the position plot, or use the horizontal endpoint slider,
to replace only the automatically detected end while keeping the start fixed.

The intermediate files are kept separately: cropped signals are stored in
`output/summaries/manual_crops/P##/`, selected detection windows in
`manual_detection_windows_P##.csv`, and successful synchronization results in
`manual_sync_report_P##.csv`. A complete existing manual synchronization report
is checked before processing starts; the workflow asks whether it should be
reused.

To open the manual interface independently:

```text
python -m emg_force.manual_synchronisation --data /path/to/data --participant 3
```

To explicitly reuse a saved manual report in the main workflow:

```text
python -m emg_force.cli --manual-sync-report /path/to/output/manual_sync_report_P03.csv
```

There is no automatic synchronization fallback in the main workflow.

The raw-data viewer can be run with:

```text
python -m emg_force.raw_data_viewer
```

## Study conditions

- Modality: vertical (top-rope) and horizontal (lateral bouldering)
- BWS levels: 25%, 50%, 75% of body weight (if feasible with patient)
- EMG: biceps brachii, triceps, supraspinatus, flexor carpi ulnaris, erector spinae
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
