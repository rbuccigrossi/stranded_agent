import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT = PROJECT_ROOT / "skills" / "file-tool" / "scripts" / "file_tool.py"


class FileToolTests(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], check=True,
                              capture_output=True, text=True)

    def test_insert_preserves_line_endings(self):
        cases = [(b"a\nb\n", b"a\nX\nb\n"), (b"a\r\nb\r\n", b"a\r\nX\r\nb\r\n")]
        with tempfile.TemporaryDirectory() as directory:
            for index, (source, expected) in enumerate(cases):
                path = Path(directory) / f"insert-{index}.txt"
                path.write_bytes(source)
                self.run_tool("insert-lines", str(path), "2", "X")
                self.assertEqual(path.read_bytes(), expected)

    def test_edit_preserves_unterminated_final_line(self):
        cases = [(b"a\nb", b"a\nX"), (b"a\r\nb", b"a\r\nX")]
        with tempfile.TemporaryDirectory() as directory:
            for index, (source, expected) in enumerate(cases):
                path = Path(directory) / f"edit-{index}.txt"
                path.write_bytes(source)
                self.run_tool("edit-lines", str(path), "2", "2", "X")
                self.assertEqual(path.read_bytes(), expected)

    def test_skill_has_standard_frontmatter(self):
        skill = (PROJECT_ROOT / "skills" / "file-tool" / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(skill.startswith("---\n"))
        self.assertIn("name: file-tool", skill)
        self.assertIn("description:", skill)


if __name__ == "__main__":
    unittest.main()
