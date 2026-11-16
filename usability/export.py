"""Export extracted questionnaire answers to tables and a readable readout."""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .questionnaire_schema import QUESTION_ORDER
from .simple_xlsx import write_workbook


OUTPUT_COLUMNS = [
    "participant", "section", "question_id", "question", "condition", "answer",
    "answer_type", "source_file", "field_name", "order",
]


def records_dataframe(records):
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def _dataframe_rows(dataframe):
    """Convert a DataFrame to rows for the dependency-free XLSX fallback."""
    values = dataframe.where(pd.notna(dataframe), "").astype(object).values.tolist()
    return [list(dataframe.columns)] + values


def overview_dataframe(records):
    """Create one row per question/condition and one column per participant."""
    long = records_dataframe(records)
    if long.empty:
        return pd.DataFrame(columns=["section", "question_id", "question", "condition", "answer_type"])

    index_columns = ["section", "question_id", "question", "condition", "answer_type"]
    table = long.pivot_table(
        index=index_columns,
        columns="participant",
        values="answer",
        aggfunc=lambda values: "; ".join(str(value) for value in values if str(value).strip()),
        fill_value="",
    ).reset_index()
    table.columns.name = None
    table["_question_order"] = table["question_id"].map(QUESTION_ORDER).fillna(9999)
    table = table.sort_values(["_question_order", "condition"], kind="stable").drop(columns="_question_order")
    return table


def readout_text(records):
    """Return a human-readable participant-by-participant summary."""
    grouped = defaultdict(list)
    for record in sorted(
        records,
        key=lambda item: (
            item["participant"],
            QUESTION_ORDER.get(item["question_id"], 9999),
            item["condition"],
            item["order"],
        ),
    ):
        grouped[record["participant"]].append(record)

    lines = []
    for participant in sorted(grouped):
        lines.append(f"Participant {participant}")
        lines.append("")
        for record in grouped[participant]:
            condition = f" [{record['condition']}]" if record["condition"] else ""
            answer = record["answer"] or "[keine Antwort]"
            lines.append(f"{record['question_id']}{condition} – {record['question']}")
            lines.append(f"Answer: {answer}")
            lines.append("")
        lines.append("=" * 72)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n" if lines else ""


def write_outputs(records, warnings, output_dir: str | Path):
    """Write CSV, Excel, and text outputs and return their paths.

    The workbook is rebuilt from all records on every run. Because the CLI
    normally scans every questionnaire document, adding a new participant and
    rerunning it updates the aggregate workbook without duplicating answers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        Alignment = Font = get_column_letter = None
    long = records_dataframe(records)
    overview = overview_dataframe(records)
    warning_table = pd.DataFrame({"warning": warnings})
    readout = readout_text(records)

    paths = {
        "long_csv": output_dir / "usability_questionnaire_long.csv",
        "overview_csv": output_dir / "usability_questionnaire_overview.csv",
        "warnings_csv": output_dir / "usability_questionnaire_warnings.csv",
        "readout_txt": output_dir / "usability_questionnaire_readout.txt",
        "workbook": output_dir / "usability_questionnaire_overview.xlsx",
    }
    long.to_csv(paths["long_csv"], index=False)
    overview.to_csv(paths["overview_csv"], index=False)
    warning_table.to_csv(paths["warnings_csv"], index=False)
    paths["readout_txt"].write_text(readout, encoding="utf-8")

    readout_table = pd.DataFrame({"readout": readout.splitlines()})
    if Alignment is None:
        write_workbook(paths["workbook"], {
            "Question overview": _dataframe_rows(overview),
            "Long format": _dataframe_rows(long),
            "Warnings": _dataframe_rows(warning_table),
            "Readout": _dataframe_rows(readout_table),
        })
    else:
        with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
            overview.to_excel(writer, sheet_name="Question overview", index=False)
            long.to_excel(writer, sheet_name="Long format", index=False)
            warning_table.to_excel(writer, sheet_name="Warnings", index=False)
            readout_table.to_excel(writer, sheet_name="Readout", index=False)

            for worksheet in writer.book.worksheets:
                worksheet.freeze_panes = "A2"
                worksheet.sheet_view.showGridLines = False
                if worksheet.max_row and worksheet.max_column:
                    worksheet.auto_filter.ref = worksheet.dimensions
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)
                for column_number, column_cells in enumerate(worksheet.columns, start=1):
                    values = [str(cell.value or "") for cell in column_cells]
                    width = min(max(len(value) for value in values) + 2, 70)
                    worksheet.column_dimensions[get_column_letter(column_number)].width = max(width, 12)
                for row in worksheet.iter_rows():
                    for cell in row:
                        cell.alignment = Alignment(vertical="top", wrap_text=True)
    return paths
