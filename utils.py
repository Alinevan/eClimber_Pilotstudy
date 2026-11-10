"""Small helpers shared by several modules."""
import re
from scipy.signal import butter, filtfilt
import config as cfg


def lowpass(x, fc, fs):
    b, a = butter(cfg.FILTER_ORDER, fc / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def clean_label(label: str) -> str:
    """Strip whitespace and a generic 'Voltage.' / 'EMG:' prefix."""
    label = re.sub(r"^(voltage|analog|emg)\s*[.:]\s*", "", label.strip(), flags=re.IGNORECASE)
    return label or "unnamed"