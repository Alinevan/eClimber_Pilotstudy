"""Shared filesystem helpers for the study analyses."""

from pathlib import Path


# Main study-data folder. Change this one line if the data folder is moved.
STUDY_DATA_ROOT = Path(
    "/Users/alinevandebroek/Documents/Career/ETH/Unterlagen_Studium/"
    "MSc_BiomedEng/HS26_Master_Thesis_BFH/Data_Analysis/Data"
)


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    """Return the configured study-data folder, validating that it exists.

    ``explicit`` is an optional override for another computer or a temporary
    test folder. Normal runs use ``STUDY_DATA_ROOT`` automatically.
    """
    root = Path(explicit).expanduser().resolve() if explicit else STUDY_DATA_ROOT
    if not root.is_dir():
        raise SystemExit(
            f"Study data folder not found: {root}\n"
            "Update common/paths.py or pass an explicit data-folder override."
        )
    return root


def find_subfolder(root: Path, keyword: str) -> Path | None:
    """Return the first direct subfolder whose name contains ``keyword``."""
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and keyword.lower() in directory.name.lower():
            return directory
    return None


def output_directory(data_root: Path, folder_name: str = "output") -> Path:
    """Return the generated-output directory next to the raw data folder."""
    directory = data_root.parent / folder_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_subdirectory(data_root: Path, subfolder: str, folder_name: str = "output") -> Path:
    """Return a named generated-output subfolder, creating it if needed."""
    directory = output_directory(data_root, folder_name) / subfolder
    directory.mkdir(parents=True, exist_ok=True)
    return directory
