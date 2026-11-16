import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from usability.docx_reader import read_docx


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _docx_xml():
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{W}">
  <w:body>
    <w:p>
      <w:r><w:t>A6. Choose a level? </w:t></w:r>
      <w:r><w:fldChar w:fldCharType="begin"><w:ffData><w:name w:val="Check1"/><w:checkBox><w:default w:val="1"/></w:checkBox></w:ffData></w:fldChar></w:r>
      <w:r><w:fldChar w:fldCharType="separate"/></w:r>
      <w:r><w:fldChar w:fldCharType="end"/></w:r>
      <w:r><w:t>25%</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:t>E2. Text answer: </w:t></w:r>
      <w:r><w:fldChar w:fldCharType="begin"><w:ffData><w:name w:val="Text2"/><w:textInput/></w:ffData></w:fldChar></w:r>
      <w:r><w:fldChar w:fldCharType="separate"/></w:r>
      <w:r><w:t>participant text</w:t></w:r>
      <w:r><w:fldChar w:fldCharType="end"/></w:r>
    </w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>ID</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Question</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
  </w:body>
</w:document>'''


class DocxReaderTests(unittest.TestCase):
    def test_reads_legacy_checkbox_and_text_field(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "questionnaire_P02.docx"
            with ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", _docx_xml())

            document = read_docx(path)
            self.assertEqual(len(document.paragraphs), 2)
            self.assertEqual(document.paragraphs[0].fields[0].checked, True)
            self.assertEqual(document.paragraphs[1].fields[0].value, "participant text")
            self.assertEqual(document.tables[0].rows[0], ["ID", "Question"])


if __name__ == "__main__":
    unittest.main()
