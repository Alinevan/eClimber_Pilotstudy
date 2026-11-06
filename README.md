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
