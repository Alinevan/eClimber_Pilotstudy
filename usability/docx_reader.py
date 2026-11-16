"""Low-level reader for the Word XML used by the questionnaire files.

The questionnaires use legacy Word form fields. Their checked state is stored
in the DOCX XML, so reading only visible paragraph text would lose answers.
This module intentionally uses the standard library instead of depending on a
particular Word installation or on ``python-docx`` exposing form fields.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union
from zipfile import BadZipFile, ZipFile
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"


def _clean_text(value: str) -> str:
    """Collapse Word's layout whitespace while preserving the answer text."""
    return " ".join(value.replace("\xa0", " ").split())


def _attribute(element, name: str, default=None):
    return element.attrib.get(W + name, default)


@dataclass
class Field:
    """One legacy Word form field."""

    name: str
    kind: str
    checked: bool | None = None
    value: str = ""


@dataclass
class Paragraph:
    text: str
    fields: list[Field] = field(default_factory=list)


@dataclass
class Table:
    rows: list[list[str]]


Block = Union[Paragraph, Table]


@dataclass
class Document:
    path: Path
    blocks: list[Block]

    @property
    def paragraphs(self) -> list[Paragraph]:
        return [block for block in self.blocks if isinstance(block, Paragraph)]

    @property
    def tables(self) -> list[Table]:
        return [block for block in self.blocks if isinstance(block, Table)]


def _field_from_begin(fld_char) -> Field:
    ff_data = fld_char.find(f"{W}ffData")
    if ff_data is None:
        return Field(name="", kind="unknown")

    name_element = ff_data.find(f"{W}name")
    name = _attribute(name_element, "val", "") if name_element is not None else ""

    checkbox = ff_data.find(f"{W}checkBox")
    if checkbox is not None:
        default = checkbox.find(f"{W}default")
        checked_value = _attribute(default, "val", "0") if default is not None else "0"
        return Field(name=name, kind="checkbox", checked=checked_value in {"1", "true", "on"})

    if ff_data.find(f"{W}textInput") is not None:
        return Field(name=name, kind="text")

    return Field(name=name, kind="unknown")


def _parse_paragraph(element) -> Paragraph:
    visible = []
    fields = []
    active_field = None
    field_result_started = False

    for node in element.iter():
        if node.tag == f"{W}fldChar":
            field_type = _attribute(node, "fldCharType")
            if field_type == "begin":
                active_field = _field_from_begin(node)
                fields.append(active_field)
                field_result_started = False
            elif field_type == "separate":
                field_result_started = True
            elif field_type == "end":
                active_field = None
                field_result_started = False
        elif node.tag == f"{W}t" and node.text:
            if active_field is not None and active_field.kind == "text" and field_result_started:
                active_field.value += node.text
            elif active_field is None or active_field.kind != "text":
                visible.append(node.text)
        elif node.tag == f"{W}tab":
            visible.append("\t")
        elif node.tag == f"{W}br":
            visible.append("\n")

    for item in fields:
        item.value = _clean_text(item.value)
    return Paragraph(text=_clean_text("".join(visible)), fields=fields)


def _parse_table(element) -> Table:
    rows = []
    for row_element in element.findall(f"{W}tr"):
        row = []
        for cell_element in row_element.findall(f"{W}tc"):
            text = "".join(node.text or "" for node in cell_element.iter(f"{W}t"))
            row.append(_clean_text(text))
        rows.append(row)
    return Table(rows=rows)


def read_docx(path: str | Path) -> Document:
    """Read paragraphs, tables, checkbox fields, and text fields from a DOCX."""
    path = Path(path)
    try:
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
    except (BadZipFile, KeyError, ET.ParseError) as exc:
        raise ValueError(f"Could not read Word document {path}: {exc}") from exc

    body = root.find(f"{W}body")
    if body is None:
        raise ValueError(f"Word document has no document body: {path}")

    blocks = []
    for element in body:
        if element.tag == f"{W}p":
            blocks.append(_parse_paragraph(element))
        elif element.tag == f"{W}tbl":
            blocks.append(_parse_table(element))
    return Document(path=path, blocks=blocks)
