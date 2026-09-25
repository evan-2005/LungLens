"""Unit tests for the machine-independent split manifest."""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")

from lung_attention.split_manifest import (  # noqa: E402
    fingerprint,
    read_manifest,
    relative_posix,
    resolve,
    rows_from_split,
    write_manifest,
)


class RelativePathTest(unittest.TestCase):
    def test_round_trip_uses_forward_slashes(self):
        root = os.path.join("data", "tb")
        path = os.path.join(root, "Normal", "Normal-1.png")
        rel = relative_posix(path, root)
        self.assertEqual(rel, "Normal/Normal-1.png")
        self.assertEqual(os.path.normpath(resolve(rel, root)), os.path.normpath(path))

    def test_path_outside_root_is_rejected(self):
        with self.assertRaises(ValueError):
            relative_posix(os.path.join("elsewhere", "x.png"), os.path.join("data", "tb"))


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.roots = {"tb_ds": os.path.join("r", "tb"), "shenzhen_tb": os.path.join("r", "sz")}
        self.split = {
            "train": ([os.path.join("r", "tb", "Normal", "a.png")], [0], ["tb_ds"]),
            "val": ([os.path.join("r", "sz", "CHNCXR_0001_1.png")], [2], ["shenzhen_tb"]),
            "test": ([], [], []),
        }

    def test_fingerprint_ignores_row_order_and_machine_paths(self):
        rows = rows_from_split(self.split, self.roots)
        other_roots = {k: os.path.join("mnt", "other", v) for k, v in self.roots.items()}
        moved = {
            name: ([os.path.join("mnt", "other", p) for p in ps], ls, ss)
            for name, (ps, ls, ss) in self.split.items()
        }
        self.assertEqual(fingerprint(rows), fingerprint(list(reversed(rows_from_split(moved, other_roots)))))

    def test_fingerprint_changes_when_membership_changes(self):
        rows = rows_from_split(self.split, self.roots)
        swapped = [dict(r, split="test") if r["split"] == "val" else r for r in rows]
        self.assertNotEqual(fingerprint(rows), fingerprint(swapped))

    def test_write_then_read_restores_the_split(self):
        rows = rows_from_split(self.split, self.roots)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "split.csv")
            write_manifest(rows, path)
            back = read_manifest(path)
        self.assertEqual(fingerprint(back), fingerprint(rows))
        self.assertEqual(back[0]["label"], 0)


if __name__ == "__main__":
    unittest.main()
