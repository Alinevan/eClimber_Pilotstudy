"""Minimal XLSX writer used when openpyxl is not installed.

The normal path uses pandas/openpyxl for formatting. This fallback keeps the
Excel deliverable available in restricted environments without downloading a
package: it writes plain-text worksheet cells into a valid XLSX container.
"""

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_name(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _text(value) -> str:
    if value is None:
        return ""
    value = str(value)
    return "" if value.lower() == "nan" else value


def _worksheet_xml(rows) -> str:
    xml_rows = []
    for row_number, row in enumerate(rows, start=1):
        cells = []
        for column_number, value in enumerate(row, start=1):
            cell_ref = f"{_column_name(column_number)}{row_number}"
            value_xml = escape(_text(value))
            cells.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">'
                f"{value_xml}</t></is></c>"
            )
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    return (
        f'<worksheet xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def _content_types(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        f'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(overrides)}</Types>'
    )


def write_workbook(path: str | Path, sheets: dict[str, list[list[object]]]):
    """Write a basic multi-sheet workbook from ordered rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_items = list(sheets.items())

    workbook_sheets = []
    workbook_relationships = []
    for index, (name, _rows) in enumerate(sheet_items, start=1):
        workbook_sheets.append(f'<sheet name="{escape(name[:31])}" sheetId="{index}" r:id="rId{index}"/>')
        workbook_relationships.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )

    workbook_xml = (
        f'<workbook xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
        f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>'
    )
    workbook_rels_xml = (
        f'<Relationships xmlns="{PKG_REL_NS}">{"".join(workbook_relationships)}</Relationships>'
    )
    package_rels_xml = (
        f'<Relationships xmlns="{PKG_REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types(len(sheet_items)))
        archive.writestr("_rels/.rels", package_rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        for index, (_name, rows) in enumerate(sheet_items, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(rows))
