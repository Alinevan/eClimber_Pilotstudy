# eClimber EMG and Force Analysis

Python code for my Master's thesis: Assistive Load and Muscle Demand During Supported Climbing

## What the code does
1. Loads surface EMG data (.c3d) and eClimber force data (.csv)
2. Pre-processes the EMG (filtering, rectification, smoothing)
3. Aligns EMG and force data in time using the sync peaks in the motor RPM channel
4. Plots EMG and force for each condition
5. Calculates the outcome values for the study:
   - Muscle activation per muscle, modality (vertical/horizontal) and BWS level (25/50/75%)
   - BWS force: mean, peak, SD, and ratio of measured to nominal force

## Manual synchronisation

For trials where the force recording is shifted and the fixed first-seconds
search window does not contain the sync peaks, install the requirements and
run:

```text
python manual_synchronisation.py --data /path/to/data --participant 3
```

The browser view shows the filtered left/right biceps EMG envelopes and all
numeric raw force channels for each condition. You can crop the displayed
filtered EMG and raw force independently; the raw source files are never
modified. The cropped derived files can be saved under
`output/manual_crops/P03/`.

The synchronization mode defaults to position valleys when a position channel
is available. Use Plotly's box-select tool to drag over the cropped force trace
and choose the detection window. The tool finds the three most prominent
valleys (or peaks if legacy mode is selected) and matches them to the EMG
peaks. Clicking “Detect peaks and synchronize” overlays the extrema, shifts
the force time axis for visual inspection, and saves the result to
`output/manual_sync_report_P03.csv`.

To use those manually selected offsets in the regular analysis outputs, run:

```text
python main.py --manual-sync-report /path/to/output/manual_sync_report_P03.csv
```

The original fixed-window batch synchronisation remains the default when this
option is omitted and is available through `synchronisation.synchronize()`.

## Study conditions
- Modality: vertical (top-rope) and horizontal (lateral bouldering)
- BWS levels: 25%, 50%, 75% of body weight (if feasible with patient)
- EMG: biceps brachii, triceps, infraspinatus, flexor ulnaris, erector spinae
- No MVC normalization

## How to run
1. Install Python 3.11 and the packages in `requirements.txt`
2. Put your data files in a local `data/` folder (not uploaded to GitHub)
3. Run `python plot_climbing_emg_force.py`

## Data
Data files are not part of this repository because they contain participant data. Please adjust the data path accordingly in the config.py file.

## Author
Aline Vandebroek, 2026
