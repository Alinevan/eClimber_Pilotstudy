from pathlib import Path

# Paths
REPO_DIR = Path(__file__).resolve().parent
EMG_DATA_DIR = REPO_DIR.parent.parent / "Data" / "EMG"
FORCE_DATA_DIR = REPO_DIR.parent.parent / "Data" / "Force"
OUTPUT_DIR = REPO_DIR.parent.parent / "Analysis" / "figures"

# EMG settings
EMG_SAMPLING_RATE =         # Hz
BANDPASS_LOW =                # Hz
BANDPASS_HIGH =              # Hz
RMS_WINDOW_MS =             # ms

# Muscles
MUSCLES = ["biceps_brachii", "triceps_brachii", "infraspinatus", "flexor_carpi_ulnaris", "erector_spinae"]

# Study conditions
MODALITIES = ["vertical", "horizontal"]
BWS_LEVELS = [25, 50, 75]       # % of body weight

# Synchronisation
N_SYNC_PEAKS = 3