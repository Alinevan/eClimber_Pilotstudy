"""Settings specific to the EMG/force analysis pipeline."""

# --- EMG processing ---
BANDPASS_HZ = (20.0, 450.0)    # EMG band-pass
ENVELOPE_LP_HZ = 10.0          # linear envelope low-pass cut-off
FILTER_ORDER = 4               # Butterworth order (zero-phase with filtfilt)
PLOT_FS_HZ = 100.0             # envelopes are decimated to this rate for plotting

# --- Force ---
FORCE_FS_DEFAULT = 1000.0      # used only if a force CSV has no time column
FORCE_LP_HZ = None             # e.g. 10.0 to low-pass the force; None = raw
FORCE_OUTCOME_COLUMN = "Force Actual__N"  # measured rope force for force outcomes
NOMINAL_FORCE_FILENAME = "Nominal_F_Values.xlsx"
NOMINAL_FORCE_KG_TO_N = 10.0   # eClimber protocol: 10 N per kilogram setting

# --- EMG outcomes ---
NORMALIZATION_MOVING_AVERAGE_S = 0.5  # paper's participant/muscle reference window

# --- File naming ---
# Condition = keyword at the START of the file name (or after "_"/"-"), followed by the BWS level,
# e.g. "Lat50_P01.csv", "Vert_75_P01.c3d", "Horizontal_25_P03.c3d". Case-insensitive.
# Keyword -> climbing direction ("lateral"/"lat" are treated as horizontal)
DIRECTION_ALIASES = {"horizontal": "horizontal", "hor": "horizontal",
                     "lateral": "horizontal", "lat": "horizontal",
                     "vertical": "vertical", "vert": "vertical"}
# BWS levels expected per direction; missing levels are reported in the output
EXPECTED_BWS = (25, 50, 75)

# Participant ID = "_P" + number at the END of the file name (before the extension),
# e.g. "Pilot_Horizontal_25_P03.c3d" -> participant 3
PARTICIPANT_PREFIX = "P"
# Subfolders of the data folder are found by keyword in their name (case-insensitive),
# e.g. "EMG", "emg_data" -> EMG;  "Force", "Force_data" -> force
EMG_FOLDER_KEYWORD = "emg"
FORCE_FOLDER_KEYWORD = "force"

# --- EMG channels (as labelled in the C3D) ---
EMG_LABELS = ["L.Biceps Br.", "R.Biceps Br.", "L.Med. Triceps", "R.Med. Triceps",
              "L.Infraspin.", "R.Infraspin.", "L.Flex.Carp.R", "R.Flex.Carp.R"]
# The C3D channel keys above are retained for loading compatibility. These are
# the anatomically corrected labels shown to users in plots.
EMG_DISPLAY_LABELS = {
    "L.Infraspin.": "L. Supraspinatus",
    "R.Infraspin.": "R. Supraspinatus",
    "L.Flex.Carp.R": "L. Flexor Carpi Ulnaris",
    "R.Flex.Carp.R": "R. Flexor Carpi Ulnaris",
}
# Used only when EMG_LABELS is empty: analog channels containing these are skipped
EMG_EXCLUDE_KEYWORDS = ["force", "trigger", "sync", "acc", "gyro", "mag", "battery", "switch", "Emg"]

# Optional: merge channels into one muscle group (averaged), e.g. {"Biceps_L": "Biceps"}
MUSCLE_GROUP_MAP = {}

# --- Synchronisation EMG <-> force ---
SYNC_EMG_LABEL = "R.Biceps Br."         # default EMG channel with the sync contractions
# SYNC_FORCE_COLUMN = "Force Actual__N"
SYNC_FORCE_COLUMN = "RPM Motor 1__RPM"  # fallback default in the manual selector
SYNC_N_PEAKS = 3
SYNC_WINDOW_S = 20.0                    # initial size of the manual EMG window
SYNC_MIN_PEAK_DISTANCE_S = 1
SYNC_EMG_MIN_PEAK_DISTANCE_S = 0.35     # EMG sync pulses may be closer than force dips
SYNC_FORCE_LP_HZ = 20.0                 # low-pass used only for force peak detection
SYNC_FORCE_INVERT = False               # used only if manual signal-peak mode is selected
CALCULATION_START_DELAY_S = 1.0         # start analysis after the final sync event
CROP_TO_EMG = True                      # crop force to the EMG recording duration

# Interactive synchronization marker colors; these are deliberately distinct
# from the red/orange EMG traces and the force/position traces.
SYNC_EMG_DETECTION_COLOR = "#8e44ad"    # purple detection trace
SYNC_EMG_PEAK_COLOR = "#39ff14"         # bright green EMG peak crosses
SYNC_POSITION_PEAK_COLOR = SYNC_EMG_PEAK_COLOR  # bright green position valley crosses

# --- Plotting ---
COLORS = {25: "#1f77b4", 50: "#ff7f0e", 75: "#2ca02c"}
FALLBACK_COLORS = ["#d62728", "#9467bd", "#8c564b", "#e377c2"]
DIRECTIONS = ("horizontal", "vertical")
OUTPUT_FOLDER = "output"         # created next to the data folder (same parent)
