"""Tests for fcp_pipeline.outline section reading."""

import unittest
from pathlib import Path

from fcp_pipeline.outline import OutlineError, Section, parse_outline, read_outline

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "outlines"


class TestOutline(unittest.TestCase):
    def test_sections_numbered_in_order(self):
        sections = read_outline(FIXTURES / "two-level.md")
        self.assertEqual([s.number for s in sections], [1, 2, 3])
        self.assertEqual(
            [s.label for s in sections],
            ["1 Cold open", "2 What I brought", "3 Evening routine"],
        )
        titles = [s.title for s in sections]
        self.assertEqual(titles, ["Cold open", "What I brought", "Evening routine"])
        self.assertNotIn("Cutting a short film", titles)

    def test_wish_lines_attached(self):
        sections = {s.id: s for s in read_outline(FIXTURES / "two-level.md")}
        self.assertEqual(
            sections["cold-open"].wishes,
            (
                "Wide shot of the doorway at dawn",
                "Close-up of hands packing a bag",
                "Reaction shot from the stairs",
            ),
        )
        self.assertEqual(
            sections["what-i-brought"].wishes,
            (
                "Folded jacket on the chair",
                "Spare batteries in a row",
                "Notebook beside the lamp",
            ),
        )
        self.assertEqual(
            sections["evening-routine"].wishes,
            (
                "Kettle steam against the window",
                "Lamp switched on",
                "Chair pulled to the table",
            ),
        )
        self.assertEqual(
            sections["cold-open"].notes,
            ("A quiet hallway before the day begins.", "### Beat note"),
        )
        self.assertNotIn(
            "A quiet hallway before the day begins.",
            sections["cold-open"].wishes,
        )
        self.assertNotIn("### Beat note", sections["cold-open"].wishes)

    def test_reorder_keeps_ids(self):
        before = {s.id: s for s in read_outline(FIXTURES / "two-level.md")}
        ordered = read_outline(FIXTURES / "reordered.md")
        after = {s.id: s for s in ordered}
        self.assertEqual(set(after), set(before))
        for section_id, first in before.items():
            with self.subTest(section_id=section_id):
                self.assertEqual(after[section_id].title, first.title)
                self.assertEqual(after[section_id].wishes, first.wishes)
        self.assertEqual(
            [s.id for s in ordered],
            ["evening-routine", "cold-open", "what-i-brought"],
        )
        self.assertEqual([s.number for s in ordered], [1, 2, 3])

    def test_empty_outline_error(self):
        with self.assertRaises(OutlineError) as raised:
            read_outline(FIXTURES / "empty.md")
        self.assertEqual(raised.exception.line, 1)
        self.assertTrue(str(raised.exception).startswith("empty.md line 1:"))

    def test_level_one_used_when_no_level_two(self):
        sections = read_outline(FIXTURES / "level-one.md")
        self.assertEqual(
            [s.title for s in sections],
            ["Cold open", "What I brought", "Evening routine"],
        )
        self.assertEqual([s.number for s in sections], [1, 2, 3])
        by_id = {s.id: s for s in sections}
        self.assertEqual(
            by_id["cold-open"].wishes,
            (
                "Wide shot of the doorway at dawn",
                "Close-up of hands packing a bag",
            ),
        )

    def test_bullet_before_heading_error(self):
        text = "- A wish with no section yet\n\n## Cold open\n- A proper wish\n"
        with self.assertRaises(OutlineError) as raised:
            parse_outline(text)
        self.assertEqual(raised.exception.line, 1)
        self.assertTrue(str(raised.exception).startswith("outline.md line 1:"))

    def test_duplicate_id_error(self):
        text = "## Cold open\n- One wish\n\n## Cold open\n- Another wish\n"
        with self.assertRaises(OutlineError) as raised:
            parse_outline(text)
        self.assertEqual(raised.exception.line, 4)
        message = str(raised.exception)
        self.assertTrue(message.startswith("outline.md line 4:"))
        self.assertIn("line 1", message)

    def test_empty_title_error(self):
        text = "# Outline title\n\n##\n- A wish\n"
        with self.assertRaises(OutlineError) as raised:
            parse_outline(text)
        self.assertEqual(raised.exception.line, 3)
        self.assertTrue(str(raised.exception).startswith("outline.md line 3:"))

    def test_numbered_heading_title(self):
        text = "## 3. Cold open\n- One wish\n\n## 10 things I packed\n- Another wish\n"
        sections = parse_outline(text)
        self.assertEqual(sections[0].title, "Cold open")
        self.assertEqual(sections[0].id, "cold-open")
        self.assertEqual(sections[1].title, "10 things I packed")
        self.assertEqual(sections[1].id, "10-things-i-packed")

    def test_section_metadata(self):
        self.assertTrue(issubclass(OutlineError, ValueError))
        section = read_outline(FIXTURES / "two-level.md")[0]
        self.assertIsInstance(section, Section)
        self.assertEqual(section.line, 3)
        self.assertIsInstance(section.wishes, tuple)
        self.assertIsInstance(section.notes, tuple)
        self.assertEqual(section.label, "1 Cold open")
        with self.assertRaises(AttributeError):
            section.number = 9


if __name__ == "__main__":
    unittest.main()
