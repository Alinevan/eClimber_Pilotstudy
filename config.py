"""All settings in one place. Other modules do `import config as cfg`."""

# --- EMG processing ---
BANDPASS_HZ = (20.0, 450.0)    # EMG band-pass
ENVELOPE_LP_HZ = 10.0          # linear envelope low-pass cut-off
FILTER_ORDER = 4               # Butterworth order (zero-phase with filtfilt)
PLOT_FS_HZ = 100.0             # envelopes are decimated to this rate for plotting

# --- Force ---
FORCE_FS_DEFAULT = 1000.0      # used only if a force CSV has no time column
FORCE_LP_HZ = None             # e.g. 10.0 to low-pass the force; None = raw

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
# Used only when EMG_LABELS is empty: analog channels containing these are skipped
EMG_EXCLUDE_KEYWORDS = ["force", "trigger", "sync", "acc", "gyro", "mag", "battery", "switch", "Emg"]

# Optional: merge channels into one muscle group (averaged), e.g. {"Biceps_L": "Biceps"}
MUSCLE_GROUP_MAP = {}

# --- Synchronisation EMG <-> force ---
SYNC_EMG_LABEL = "L.Biceps Br."         # EMG channel with the sync contractions
# SYNC_FORCE_COLUMN = "Force Actual__N"
SYNC_FORCE_COLUMN = "RPM Motor 1__RPM"  # force column with the sync peaks
SYNC_N_PEAKS = 3
SYNC_WINDOW_S = 20.0                    # look for peaks in the first N seconds
SYNC_MIN_PEAK_DISTANCE_S = 1
SYNC_FORCE_LP_HZ = 20.0                 # low-pass used only for force peak detection
SYNC_FORCE_INVERT = False               # True if the sync peaks point downwards
SYNC_MAX_SPREAD_S = 0.6                 # warn if peak offsets disagree by more than this
CROP_TO_EMG = True                      # crop force to the EMG recording duration

# --- Plotting ---
COLORS = {25: "#1f77b4", 50: "#ff7f0e", 75: "#2ca02c"}
FALLBACK_COLORS = ["#d62728", "#9467bd", "#8c564b", "#e377c2"]
DIRECTIONS = ("horizontal", "vertical")
OUTPUT_FOLDER = "output"         # created next to the data folder (same parent)