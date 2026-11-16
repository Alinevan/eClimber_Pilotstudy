import unittest
from pathlib import Path

from common.participants import parse_participant, participant_tag


class ParticipantTests(unittest.TestCase):
    def test_parse_participant_from_emg_filename(self):
        self.assertEqual(parse_participant(Path("Vertical_25_P03.c3d")), 3)

    def test_parse_participant_from_questionnaire_filename(self):
        self.assertEqual(parse_participant("Usability Fragebogen_P02.docx"), 2)

    def test_non_matching_filename_returns_none(self):
        self.assertIsNone(parse_participant("questionnaire_template.docx"))

    def test_participant_tag(self):
        self.assertEqual(participant_tag(2), "P02")


if __name__ == "__main__":
    unittest.main()
