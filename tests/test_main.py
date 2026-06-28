import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1] / "src" / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class _CheckpointerContext:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, traceback):
        return False


class MainDiagramTests(unittest.TestCase):
    def test_main_shows_compiled_graph_before_input_loop(self):
        main = importlib.import_module("main")
        app = object()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            with patch.object(sys, "argv", ["main.py"]):
                with patch.object(main, "get_sqlite_checkpointer_cm", return_value=_CheckpointerContext()):
                    with patch.object(main, "build_graph", return_value=app):
                        with patch.object(main, "show_graph") as show_graph:
                            with patch("builtins.input", side_effect=["exit"]):
                                main.main()

        show_graph.assert_called_once_with(app)


if __name__ == "__main__":
    unittest.main()
