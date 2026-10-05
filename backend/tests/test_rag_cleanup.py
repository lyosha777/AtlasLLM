import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.evaluation import rag


class RagCleanupTests(unittest.TestCase):
    def test_cleanup_after_worker_exit_preserves_result_and_reports_locks(self):
        for status in ("completed", "error"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as root:
                joined = []
                stores = []

                class Process:
                    def __init__(self, target, args):
                        self.output = Path(args[1])
                        self.store = Path(args[2])
                        stores.append(self.store)

                    def start(self):
                        (self.store / "data_level0.bin").write_bytes(b"fixture")
                        rag.save(self.output, dict(status=status, records=[], error="Real retrieval failure" if status == "error" else None))

                    def join(self, timeout):
                        joined.append(True)

                    def is_alive(self):
                        return False

                original = rag.cleanup_store

                def cleanup(directory):
                    self.assertTrue(joined, "Store deletion must follow worker exit")
                    return original(directory)

                with patch.object(rag, "ROOT", Path(root)), \
                     patch.object(rag.multiprocessing, "get_context") as context, \
                     patch.object(rag, "cleanup_store", side_effect=cleanup), \
                     patch("sys.argv", ["rag"]), patch("builtins.print"):
                    context.return_value.Process = Process
                    self.assertEqual(rag.main(), 0 if status == "completed" else 1)
                self.assertFalse(stores[0].exists())
                result = json.loads(next((Path(root) / "data" / "benchmarks").glob("*.json")).read_text())
                self.assertEqual(result["status"], status)
                if status == "error":
                    self.assertEqual(result["error"], "Real retrieval failure")

        directory = tempfile.mkdtemp(prefix="atlas-rag-")
        try:
            with patch.object(rag.shutil, "rmtree", side_effect=PermissionError("WinError 32")) as remove, \
                 patch.object(rag.time, "sleep"):
                warning = rag.cleanup_store(directory)
                self.assertIn(directory, warning)
                self.assertEqual(remove.call_count, 3)
            with self.assertRaises(ValueError):
                rag.cleanup_store(Path(directory) / "production")
        finally:
            rag.cleanup_store(directory)


if __name__ == "__main__":
    unittest.main()
