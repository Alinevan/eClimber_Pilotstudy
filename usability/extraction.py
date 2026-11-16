"""Convert one questionnaire document into analysis-ready answer records."""

from pathlib import Path
import re

from common.participants import parse_participant, participant_tag

from .docx_reader import Document, Paragraph, Table, read_docx
from .questionnaire_schema import (
    B1_OPTIONS,
    BODY_WEIGHT_OPTIONS,
    D2_OPTIONS,
    TABLE_CONDITION_QUESTIONS,
    TABLE_QUESTIONS,
    YES_NO_UNSURE,
)


_QUESTION_RE = re.compile(r"^([A-F]\d+)\.?\s*(.*)$", re.IGNORECASE)
_SECTION_HEADINGS = {
    "Laterales Klettern": "A",
    "Klettererlebnis": "B",
    "Bedienungspanel": "C",
    "Not-Stopp": "D",
    "Ablass-Funktion": "E",
    "Selbstständigkeit (für Rollstuhl-Nutzende)": "F",
}
_REMARK_RE = re.compile(r"^\s*Bemerkung(?:en)?\s*:?(.*)$", re.IGNORECASE)
_NEEDS_RE = re.compile(r"^\s*Bedürfnis(?:se)?\s*:?(.*)$", re.IGNORECASE)
_GENERAL_RE = re.compile(r"^\s*Generelle Bemerkungen\s*:?(.*)$", re.IGNORECASE)


def _record(participant, source_file, section, question_id, question, answer,
            answer_type, condition="", order=0, field_name=""):
    return {
        "participant": participant,
        "section": section,
        "question_id": question_id,
        "question": question,
        "condition": condition,
        "answer": answer.strip(),
        "answer_type": answer_type,
        "source_file": source_file,
        "field_name": field_name,
        "order": order,
    }


def _question_id(text: str) -> str | None:
    match = _QUESTION_RE.match(text.strip())
    return match.group(1).upper() if match else None


def _section_heading(text: str) -> str | None:
    """Return the section letter for a visible section heading, if present."""
    text = re.sub(r"^[A-F](?:\.|\s)+", "", text.strip(), flags=re.IGNORECASE)
    for heading, section in _SECTION_HEADINGS.items():
        if text.casefold() == heading.casefold():
            return section
    return None


def _is_remark(text: str) -> bool:
    return _REMARK_RE.match(text.strip()) is not None


def _question_prompt(text: str) -> str:
    """Return the question text, retaining its wording but not its answer labels."""
    text = text.strip()
    match = _QUESTION_RE.match(text)
    if match:
        text = match.group(2).strip()
    question_mark = text.find("?")
    if question_mark >= 0:
        return text[:question_mark + 1]
    return text


def _checkbox_answers(paragraph: Paragraph, options: tuple[str, ...]):
    fields = [field for field in paragraph.fields if field.kind == "checkbox"]
    if len(fields) != len(options):
        return None, f"Expected {len(options)} checkboxes but found {len(fields)}"
    return [option for option, field in zip(options, fields) if field.checked], None


def _checkbox_paragraph(paragraphs, index: int) -> Paragraph:
    """Find checkbox fields in the question paragraph or its answer line."""
    paragraph = paragraphs[index]
    if any(field.kind == "checkbox" for field in paragraph.fields):
        return paragraph
    for following in paragraphs[index + 1:]:
        text = following.text.strip()
        if (_question_id(text) or _section_heading(text) or _is_remark(text)
                or _NEEDS_RE.match(text) or _GENERAL_RE.match(text)):
            break
        if any(field.kind == "checkbox" for field in following.fields):
            return following
    return paragraph


def _extract_tables(document: Document, participant: str, source_file: str, order_start: int):
    records = []
    warnings = []
    order = order_start
    for table in document.tables:
        for row in table.rows:
            if len(row) < 3:
                continue
            question_id = row[0].strip().upper()
            if question_id in TABLE_CONDITION_QUESTIONS:
                if len(row) < 5:
                    warnings.append(f"{question_id}: expected 25%, 50%, and 75% answer cells")
                    continue
                for condition, answer in zip(BODY_WEIGHT_OPTIONS[:3], row[2:5]):
                    records.append(_record(
                        participant, source_file, "A", question_id, row[1], answer,
                        "text", condition=condition, order=order,
                    ))
                    order += 1
            elif question_id in TABLE_QUESTIONS:
                section = question_id[0]
                answer_type = "free_text" if question_id == "E1" else "text"
                records.append(_record(
                    participant, source_file, section, question_id, row[1], row[2],
                    answer_type, order=order,
                ))
                order += 1
    return records, warnings, order


def _find_paragraph_index(paragraphs, question_id: str):
    return next((i for i, paragraph in enumerate(paragraphs)
                 if _question_id(paragraph.text) == question_id), None)


def _extract_direct_checkbox(paragraphs, participant, source_file, question_id,
                             options, section, order):
    index = _find_paragraph_index(paragraphs, question_id)
    if index is None:
        return [], [f"{question_id}: question not found"], order
    paragraph = _checkbox_paragraph(paragraphs, index)
    selected, warning = _checkbox_answers(paragraph, options)
    if warning:
        return [], [f"{question_id}: {warning}"], order
    return [_record(
        participant, source_file, section, question_id, _question_prompt(paragraph.text),
        "; ".join(selected) if selected else "None selected", "checkbox", order=order,
    )], [], order + 1


def _extract_d2(paragraphs, participant, source_file, order):
    index = _find_paragraph_index(paragraphs, "D2")
    if index is None:
        return [], ["D2: question not found"], order

    selected = []
    other_text = ""
    checkbox_count = 0
    for paragraph in paragraphs[index + 1:]:
        if _is_remark(paragraph.text) or paragraph.text.startswith("Generelle"):
            break
        fields = [field for field in paragraph.fields if field.kind == "checkbox"]
        if not fields:
            continue
        checkbox_count += len(fields)
        if paragraph.text.startswith("Andere"):
            if fields[0].checked:
                selected.append("Andere")
            text_fields = [field for field in paragraph.fields if field.kind == "text"]
            other_text = text_fields[0].value if text_fields else ""
        else:
            option = next((option for option in D2_OPTIONS if paragraph.text.startswith(option)), paragraph.text)
            if fields[0].checked:
                selected.append(option)

    if checkbox_count != len(D2_OPTIONS):
        warning = f"D2: expected {len(D2_OPTIONS)} checkboxes but found {checkbox_count}"
        return [], [warning], order

    answer = "; ".join(selected) if selected else "None selected"
    if other_text:
        answer += f"; Andere: {other_text}"
    record = _record(
        participant, source_file, "D", "D2", paragraphs[index].text, answer,
        "checkbox", order=order,
    )
    return [record], [], order + 1


def _extract_text_and_remarks(paragraphs, participant, source_file, records, order):
    """Add section-specific remarks and text-field answers.

    Remarks are associated with the most recent section heading or question.
    This handles a ``Bemerkungen`` field after every section without relying
    on fixed paragraph numbers or only supporting sections B and D.
    """
    warnings = []
    current_section = ""
    found_section_remarks = set()
    found_general_remarks = False
    for index, paragraph in enumerate(paragraphs):
        text = paragraph.text.strip()
        heading_section = _section_heading(text)
        if heading_section:
            current_section = heading_section

        question_id = _question_id(text)
        if question_id:
            current_section = question_id[0]

        remark_match = _REMARK_RE.match(text)
        if remark_match:
            answer_parts = [remark_match.group(1).strip()] if remark_match.group(1).strip() else []
            answer_parts.extend(field.value for field in paragraph.fields
                                if field.kind == "text" and field.value)
            # Some future templates may put the visible answer in the next
            # paragraph rather than in the same form field as the label.
            if not answer_parts:
                for following in paragraphs[index + 1:]:
                    following_text = following.text.strip()
                    if (_question_id(following_text) or _section_heading(following_text)
                            or _is_remark(following_text)
                            or _NEEDS_RE.match(following_text)
                            or _GENERAL_RE.match(following_text)):
                        break
                    answer_parts.extend(field.value for field in following.fields
                                        if field.kind == "text" and field.value)
                    if following_text:
                        answer_parts.append(following_text)
            question_id = f"{current_section}_Bemerkungen" if current_section else "Bemerkungen"
            question_label = "Bemerkungen"
            records.append(_record(
                participant, source_file, current_section, question_id, question_label,
                " ".join(answer_parts), "free_text", order=order,
            ))
            if current_section:
                found_section_remarks.add(current_section)
            order += 1
        elif _NEEDS_RE.match(text):
            needs_match = _NEEDS_RE.match(text)
            answer_parts = [needs_match.group(1).strip()] if needs_match and needs_match.group(1).strip() else []
            answer_parts.extend(field.value for field in paragraph.fields
                                if field.kind == "text" and field.value)
            found_section_remarks.add("E")
            records.append(_record(
                participant, source_file, "E", "E_Bemerkungen", "Bemerkungen",
                " ".join(answer_parts), "free_text", order=order,
            ))
            order += 1
        else:
            general_match = _GENERAL_RE.match(text)
            if not general_match:
                continue
            found_general_remarks = True
            answer_parts = [general_match.group(1).strip()] if general_match.group(1).strip() else []
            answer_parts.extend(field.value for field in paragraph.fields
                                if field.kind == "text" and field.value)
            records.append(_record(
                participant, source_file, "", "Generelle Bemerkungen", "Generelle Bemerkungen",
                " ".join(answer_parts), "free_text", order=order,
            ))
            order += 1

    # E2 and F3 are form-text fields whose visible paragraphs contain the prompt.
    for question_id, section in (("E2", "E"), ("F3", "F")):
        index = _find_paragraph_index(paragraphs, question_id)
        if index is None:
            warnings.append(f"{question_id}: question not found")
            continue
        paragraph = paragraphs[index]
        text_fields = [field for field in paragraph.fields if field.kind == "text"]
        if not text_fields:
            for following in paragraphs[index + 1:]:
                following_text = following.text.strip()
                if (_question_id(following_text) or _section_heading(following_text)
                        or _is_remark(following_text) or _NEEDS_RE.match(following_text)
                        or _GENERAL_RE.match(following_text)):
                    break
                text_fields = [field for field in following.fields if field.kind == "text"]
                if text_fields:
                    break
        value = text_fields[0].value if text_fields else ""
        if not text_fields:
            warnings.append(f"{question_id}: text field not found")
        records.append(_record(
            participant, source_file, section, question_id, _question_prompt(paragraph.text),
            value, "free_text", order=order,
            field_name=text_fields[0].name if text_fields else "",
        ))
        order += 1

    # Keep a stable row for every section, even when the participant left the
    # section's remarks field empty or the template contains no visible label.
    # This makes the aggregated overview comparable across all participants.
    for section in "ABCDEF":
        if section not in found_section_remarks:
            records.append(_record(
                participant, source_file, section, f"{section}_Bemerkungen",
                "Bemerkungen", "", "free_text", order=order,
            ))
            order += 1
    if not found_general_remarks:
        records.append(_record(
            participant, source_file, "", "Generelle Bemerkungen", "Generelle Bemerkungen",
            "", "free_text", order=order,
        ))
        order += 1
    return warnings, order


def extract_questionnaire(path: str | Path):
    """Extract one questionnaire into ``(records, warnings)``."""
    path = Path(path)
    participant_number = parse_participant(path)
    warnings = []
    if participant_number is None:
        warnings.append(f"Could not identify participant from filename: {path.name}")
        participant = path.stem
    else:
        participant = participant_tag(participant_number)

    document = read_docx(path)
    source_file = path.name
    records, table_warnings, order = _extract_tables(document, participant, source_file, 0)
    warnings.extend(table_warnings)
    paragraphs = document.paragraphs

    direct_questions = (
        ("A6", BODY_WEIGHT_OPTIONS, "A"),
        ("B1", B1_OPTIONS, "B"),
        ("D1", YES_NO_UNSURE, "D"),
        ("F1", YES_NO_UNSURE, "F"),
        ("F2", YES_NO_UNSURE, "F"),
    )
    for question_id, options, section in direct_questions:
        new_records, new_warnings, order = _extract_direct_checkbox(
            paragraphs, participant, source_file, question_id, options, section, order,
        )
        records.extend(new_records)
        warnings.extend(new_warnings)

    new_records, new_warnings, order = _extract_d2(paragraphs, participant, source_file, order)
    records.extend(new_records)
    warnings.extend(new_warnings)
    text_warnings, order = _extract_text_and_remarks(
        paragraphs, participant, source_file, records, order,
    )
    warnings.extend(text_warnings)
    return records, warnings
