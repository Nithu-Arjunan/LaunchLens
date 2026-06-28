import importlib
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "src" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"


class _GraphView:
    def draw_mermaid_png(self):
        return PNG_BYTES


class _CompiledGraph:
    def get_graph(self):
        return _GraphView()


class DiagramTests(unittest.TestCase):
    def test_show_writes_only_png_file(self):
        diagram = importlib.import_module("diagram")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "graph.png"

            result = diagram.show(_CompiledGraph(), output_path=output_path)

            self.assertEqual(result, output_path)
            self.assertEqual(output_path.read_bytes(), PNG_BYTES)
            self.assertFalse((Path(temp_dir) / "graph.html").exists())
            self.assertFalse((Path(temp_dir) / "graph.mmd").exists())

    def test_show_uses_png_suffix_for_non_png_output_path(self):
        diagram = importlib.import_module("diagram")

        with tempfile.TemporaryDirectory() as temp_dir:
            requested_path = Path(temp_dir) / "graph.html"

            result = diagram.show(_CompiledGraph(), output_path=requested_path)

            self.assertEqual(result, Path(temp_dir) / "graph.png")
            self.assertTrue(result.exists())
            self.assertFalse(requested_path.exists())


if __name__ == "__main__":
    unittest.main()
