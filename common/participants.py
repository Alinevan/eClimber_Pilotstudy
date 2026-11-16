"""Participant-ID helpers shared by EMG/force and questionnaire analyses."""

import re
from pathlib import Path


_PARTICIPANT_RE = re.compile(r"(?:^|[_\-\s])P0*(\d+)$", re.IGNORECASE)


def parse_participant(path_or_name: str | Path) -> int | None:
    """Return the participant number at the end of a filename, or ``None``.

    Examples
    --------
    ``Vertical_25_P03.c3d`` -> ``3``
    ``Usability Fragebogen_P02.docx`` -> ``2``
    """
    stem = Path(path_or_name).stem
    match = _PARTICIPANT_RE.search(stem)
    return int(match.group(1)) if match else None


def participant_tag(participant: int) -> str:
    """Return the standard participant tag used in output filenames."""
    return f"P{int(participant):02d}"
