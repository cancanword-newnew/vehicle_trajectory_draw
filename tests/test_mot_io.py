from pathlib import Path
import tempfile
import unittest

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.mot_io import MOTRow, read_mot, write_mot


class TestMOTIO(unittest.TestCase):
    def test_write_and_read_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mot.txt"
            rows = [
                MOTRow(frame=1, track_id=10, x=100.0, y=50.0, w=40.0, h=80.0, confidence=0.91),
                MOTRow(frame=2, track_id=10, x=104.0, y=53.0, w=41.0, h=79.0, confidence=0.88),
            ]

            write_mot(path, rows)
            loaded = read_mot(path)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].frame, 1)
            self.assertEqual(loaded[0].track_id, 10)
            self.assertAlmostEqual(loaded[0].confidence, 0.91, places=2)
            self.assertAlmostEqual(loaded[1].x, 104.0, places=3)

    def test_invalid_line_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            path.write_text("1,2,3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_mot(path)


if __name__ == "__main__":
    unittest.main()
