"""
Tests for TemplateManager custom template loading behavior.
"""

import tempfile
import unittest
from pathlib import Path

from openevolve.prompt.templates import TemplateManager


class TestTemplateManager(unittest.TestCase):
    """Tests for template loading from custom directories."""

    def test_hidden_sidecar_txt_files_are_ignored(self):
        """Hidden AppleDouble-style sidecar files should not break loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            (template_dir / "full_rewrite_user.txt").write_text("visible template", encoding="utf-8")
            (template_dir / "._full_rewrite_user.txt").write_bytes(
                b"\x00\x05\x16\x07binary sidecar data\xb0"
            )

            manager = TemplateManager(custom_template_dir=str(template_dir))

            self.assertEqual(manager.get_template("full_rewrite_user"), "visible template")
            self.assertNotIn("._full_rewrite_user", manager.templates)

    def test_invalid_visible_utf8_template_raises_clear_error(self):
        """A real visible template file with invalid UTF-8 should fail clearly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            bad_template = template_dir / "full_rewrite_user.txt"
            bad_template.write_bytes(b"invalid\xb0template")

            with self.assertRaisesRegex(
                ValueError,
                r"Template file '.*full_rewrite_user\.txt' must be valid UTF-8 text",
            ):
                TemplateManager(custom_template_dir=str(template_dir))


if __name__ == "__main__":
    unittest.main()
