# Analysis Notes

Working document for the eClimber EMG and force analysis.

Use of the file: notes on processing steps, aggregation of literature and basis of decisions.

## 1. Project overview

### Research question

<!-- What physiological or biomechanical question is this analysis intended to answer? -->

### Main outcomes

- Primary outcome: Muscle activity measured by EMG and load measured by the eClimber device's load cell.
- Relevant muscles: biceps brachii, triceps brachii, flexor ulnaris, supraspinatus, erector spinae
- Relevant force variables: difference between target and actual force, standard deviation

### Experimental setup

- Participant population: Individuals with incomplete spinal cord injury.
- Number of participants: 3–6 participants.
- Climbing modalities/directions: Lateral (horizontal) and vertical climbing.
- Body-weight-support levels: 25%, 50%, and 75% body-weight support.
- Electrode placement: Based on SENIAM recommendations; where SENIAM does not provide a recommendation, use scientifically supported methods from the literature.
- Sampling frequency: 1000 Hz.

## 2. Data organization and trial identification

### Input files

- EMG format: `.c3d`.
- Force format: `.csv`.
- Naming convention: `<climbing_direction>_<BWS_level>_P<participant_number>.<extension>`; for example, `Vertical_25_P01.csv` for force and `Vertical_25_P01.c3d` for EMG.
- Participant identifier: `_P<number>` at the end of the filename, for example `_P01`.
- Direction/modality labels: `vertical` and `lateral`/`horizontal`.

### Pairing rule

EMG and force files are paired using:

```text
participant + direction + BWS level
```

Example:

```text
Vertical_25_P01.c3d  <->  Vertical_25_P01.csv
```

### Data-quality checks

- [ ] All expected EMG files are present.
- [ ] All expected force files are present.
- [ ] Every filename is parsed correctly.
- [ ] Each EMG trial has the corresponding force trial.
- [ ] No duplicate files map to the same participant/direction/BWS combination.
- [ ] Missing or excluded trials are documented below.


## 3. Literature review

### Literature notes by topic

#### EMG preprocessing

<!-- Summarize recommendations for detrending, band-pass filtering, rectification, smoothing, normalization, and artifact handling. -->
- replacing NaN to 0
- removing mean and therefore DC offset
- band pass filtering: upper and lower limit, filter type, filter grade
- FiltFilt for zero phase shift
- Full wave rectification
- envelope creation: 4th order low-pass Butterworth filter at 10 Hz after rectification

#### EMG outcome measures

<!-- Summarize how RMS, iEMG, mean/envelope amplitude, and duration-normalized measures are defined and interpreted. -->

#### Synchronization

<!-- Summarize the recommended synchronization signal, peak detection, timing accuracy, and validation approach. -->

#### Force and body-weight support

<!-- Summarize how support force should be quantified and how the nominal BWS level should be interpreted. -->

### How the literature informed the setup

For each important choice, record the chain of reasoning:

```text
Literature finding -> practical constraint -> chosen method -> expected consequence
```

Examples to complete:

- Band-pass filter:
- Envelope cutoff:
- RMS definition:
- iEMG definition:
- Synchronization signal:
- Climbing-only analysis window:
- Normalization:
- Handling of missing/failed trials:

## 4. Processing pipeline

Current conceptual order:

```text
Load raw data
    -> identify participant, direction, and BWS
    -> preprocess EMG
    -> detect synchronization peaks
    -> align force time to EMG time
    -> define climbing analysis window
    -> calculate EMG outcomes for climbing only
    -> calculate force outcomes
    -> generate plots and reports
```

### Software and versions

- Python version:
- Environment:
- Key packages:
- Repository commit:
- Date of analysis:

### Code modules

| Module | Responsibility |
|---|---|
| `config.py` | Analysis parameters and labels |
| `data_loading.py` | File discovery, parsing, and raw-data loading |
| `emg_processing.py` | EMG filtering and feature calculation |
| `synchronisation.py` | EMG/force time alignment |
| `plotting.py` | Interactive HTML figures |
| `output_calculations.py` | Summary tables and saved outputs |
| `main.py` | Analysis workflow |

## 5. EMG preprocessing decisions

### Current/default settings

| Processing step | Setting | Rationale/reference | Decision status |
|---|---|---|---|
| Demeaning | Subtract channel mean |  |  |
| Missing values | Replace with zero after demeaning |  |  |
| Band-pass filter | 20–450 Hz |  |  |
| Filter type/order | Butterworth, order 4 |  |  |
| Phase handling | Zero-phase `filtfilt` |  |  |
| Rectification | Absolute value |  |  |
| Envelope filter | Low-pass, 10 Hz |  |  |
| Plotting rate | 100 Hz envelope |  |  |
| Normalization | None currently |  |  |

### EMG channels

List the included channels and document any exclusions or label corrections:

| Original label | Analysis label | Included? | Reason |
|---|---|---|---|
|  |  |  |  |

### Artifact handling

- Power-line noise:
- Movement artifacts:
- Electrode artifacts:
- Synchronization contractions:
- Missing samples:
- Rejected channels/trials:

## 6. Synchronization decisions

### Current method

- EMG synchronization channel:
- Force-side synchronization channel:
- Number of peaks:
- Search window:
- Minimum peak distance:
- Force filtering for peak detection:
- Peak polarity:
- Maximum allowed peak-offset spread:
- Force cropped to EMG duration?:

### Validation

- [ ] Each condition has the expected number of synchronization peaks.
- [ ] EMG and force peaks are visually aligned.
- [ ] The calculated offset is plausible.
- [ ] Peak-offset spread is below the chosen limit.
- [ ] Failed synchronization is documented in the output report.

Notes on failed or questionable synchronization:

<!-- Record direction, BWS, file names, reason, and action taken. -->

## 7. Defining the climbing-only analysis window

The synchronization contractions are calibration events and must not contribute to muscle-load outcomes.

### Planned rule

Recommended starting rule:

```text
climbing_start = end of the third synchronization contraction + safety margin
climbing_end   = end of the climbing trial, excluding post-trial rest
```

Document the chosen implementation:

- How is the end of the third pulse identified?
- Safety margin:
- How is climbing end identified?
- Are start/end times condition-specific?
- Are the same times applied to every EMG channel?
- How are uncertain trials handled?

### Analysis-window table

| Participant | Direction | BWS | Start (s) | End (s) | Duration (s) | Method | Quality note |
|---|---|---:|---:|---:|---:|---|---|
|  |  |  |  |  |  |  |  |

## 8. Outcome definitions

### RMS

Proposed definition:

```text
RMS = sqrt(mean(filtered_EMG_climbing_window²))
```

Record whether RMS is calculated from:

- [ ] Band-pass-filtered EMG
- [ ] Rectified EMG
- [ ] EMG envelope

Chosen definition and rationale:

### Integrated EMG / area under the curve

Proposed definition:

```text
iEMG = integral of the rectified/enveloped EMG over the climbing window
```

Integration method:

- [ ] Simpson integration
- [ ] Trapezoidal integration

Because iEMG depends on trial duration, also consider reporting:

```text
iEMG per second = iEMG / climbing-window duration
```

Chosen metric(s) and rationale:

### Force outcomes

- Force channel(s):
- Mean force:
- Peak force:
- Standard deviation:
- Measured/nominal BWS ratio:
- Force analysis window:

## 9. Decision log

Record decisions that affect the analysis, including rejected alternatives.

| Date | Decision | Alternatives considered | Reason/evidence | Code/configuration change | Reviewer/question |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## 10. Open questions and risks

- [ ] Are the synchronization contractions identical across all conditions?
- [ ] Is the third synchronization peak always before climbing starts?
- [ ] Is a fixed post-pulse safety margin appropriate?
- [ ] Is there post-climbing rest in the recordings?
- [ ] Should the climbing end be manually annotated or detected automatically?
- [ ] Are EMG amplitude comparisons valid without MVC normalization?
- [ ] Are trial durations comparable enough for iEMG comparison?
- [ ] How should failed synchronization trials be reported?
- [ ] Are there duplicate or ambiguously named files?

## 11. Reproducibility checklist

- [ ] Repository synchronized with GitHub.
- [ ] Correct Python environment activated.
- [ ] Input data location recorded.
- [ ] Configuration values recorded.
- [ ] File-pairing report reviewed.
- [ ] Synchronization report reviewed.
- [ ] Analysis windows reviewed.
- [ ] Failed trials documented.
- [ ] Generated HTML plots reviewed.
- [ ] Summary CSV files reviewed.
- [ ] Final repository commit recorded.

## 12. Analysis diary

Use this section as a chronological log.

### YYYY-MM-DD

- What I changed:
- Why I changed it:
- What I observed:
- What I still need to check:
- Relevant files/commit:

