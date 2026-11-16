import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from usability.docx_reader import Paragraph
from usability.export import overview_dataframe, readout_text, write_outputs
from usability.extraction import _extract_text_and_remarks


class QuestionnaireExportTests(unittest.TestCase):
    def test_remarks_are_assigned_to_the_current_section(self):
        paragraphs = [
            Paragraph("Laterales Klettern"),
            Paragraph("A6. Question"),
            Paragraph("Bemerkungen: A note"),
            Paragraph("Klettererlebnis"),
            Paragraph("B1. Question"),
            Paragraph("Bemerkungen: B note"),
            Paragraph("Bedienungspanel"),
            Paragraph("Bemerkungen: C note"),
        ]
        records = []
        warnings, _ = _extract_text_and_remarks(
            paragraphs, "P02", "questionnaire_P02.docx", records, 0,
        )
        self.assertEqual(
            [(record["question_id"], record["section"], record["answer"]) for record in records[:3]],
            [
                ("A_Bemerkungen", "A", "A note"),
                ("B_Bemerkungen", "B", "B note"),
                ("C_Bemerkungen", "C", "C note"),
            ],
        )
        self.assertEqual(
            {record["question_id"] for record in records if record["question_id"].endswith("_Bemerkungen")},
            {"A_Bemerkungen", "B_Bemerkungen", "C_Bemerkungen", "D_Bemerkungen", "E_Bemerkungen", "F_Bemerkungen"},
        )
        self.assertTrue(any("E2" in warning for warning in warnings))

    def test_overview_keeps_conditions_separate(self):
        records = [
            {
                "participant": "P02", "section": "A", "question_id": "A1",
                "question": "Question", "condition": "25%", "answer": "nein",
                "answer_type": "text", "source_file": "P02.docx", "field_name": "", "order": 0,
            },
            {
                "participant": "P02", "section": "A", "question_id": "A1",
                "question": "Question", "condition": "50%", "answer": "ja",
                "answer_type": "text", "source_file": "P02.docx", "field_name": "", "order": 1,
            },
        ]
        overview = overview_dataframe(records)
        self.assertEqual(list(overview["condition"]), ["25%", "50%"])
        self.assertEqual(list(overview["P02"]), ["nein", "ja"])

    def test_readout_includes_body_weight_condition(self):
        records = [{
            "participant": "P02", "section": "A", "question_id": "A1",
            "question": "Question", "condition": "75%", "answer": "ja",
            "answer_type": "text", "source_file": "P02.docx", "field_name": "", "order": 0,
        }]
        text = readout_text(records)
        self.assertIn("A1 [75%]", text)
        self.assertIn("Answer: ja", text)

    def test_excel_workbook_is_created(self):
        records = [{
            "participant": "P02", "section": "A", "question_id": "A1",
            "question": "Question", "condition": "25%", "answer": "nein",
            "answer_type": "text", "source_file": "P02.docx", "field_name": "", "order": 0,
        }]
        with TemporaryDirectory() as directory:
            paths = write_outputs(records, [], directory)
            workbook = Path(paths["workbook"])
            self.assertTrue(workbook.exists())
            with ZipFile(workbook) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("xl/worksheets/sheet1.xml", archive.namelist())


if __name__ == "__main__":
    unittest.main()
