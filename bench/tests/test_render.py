import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RENDER_PATH = Path(__file__).resolve().parents[1] / "runner" / "render.py"
_render_spec = importlib.util.spec_from_file_location(
    "render_for_test", RENDER_PATH)
render = importlib.util.module_from_spec(_render_spec)
_render_spec.loader.exec_module(render)


class RenderTests(unittest.TestCase):
    def test_disabled_readme_render_preserves_focused_landing_page(self):
        with tempfile.TemporaryDirectory() as directory:
            readme = Path(directory, "README.md")
            focused = "# Focused landing page\n"
            readme.write_text(focused)

            with mock.patch.object(render, "README", str(readme)):
                result = render.splice_readme(
                    "generated performance", "generated quality", enabled=False)

            self.assertEqual(result, focused)
            self.assertEqual(readme.read_text(), focused)


if __name__ == "__main__":
    unittest.main()
