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


class SplitOptionsTest(unittest.TestCase):
    def setUp(self):
        from lung_attention.split_manifest import apply_split_options
        self.apply = apply_split_options
        p = lambda *parts: os.path.join("r", *parts)  # noqa: E731
        self.split = {
            "train": ([p("Lung_Opacity", "images", "a.png"), p("Normal", "images", "b.png"),
                       p("sz", "c.png")], [1, 0, 2], ["radiography_db", "radiography_db", "shenzhen_tb"]),
            "val": ([p("sz", "d.png")], [0], ["shenzhen_tb"]),
            "test": ([p("Lung_Opacity", "images", "e.png"), p("tb", "f.png")], [1, 2],
                     ["radiography_db", "tb_ds"]),
        }

    def test_no_options_returns_split_unchanged(self):
        self.assertEqual(self.apply(self.split), self.split)

    def test_drop_lung_opacity_removes_it_everywhere(self):
        out = self.apply(self.split, drop_lung_opacity=True)
        for paths, _, _ in out.values():
            self.assertFalse(any("Lung_Opacity" in x for x in paths))
        self.assertEqual(len(out["train"][0]), 2)
        self.assertEqual(len(out["test"][0]), 1)

    def test_holdout_source_tests_on_all_of_it_and_trains_on_none(self):
        out = self.apply(self.split, holdout_source="shenzhen_tb")
        self.assertEqual(sorted(out["test"][2]), ["shenzhen_tb", "shenzhen_tb"])
        self.assertNotIn("shenzhen_tb", out["train"][2] + out["val"][2])

    def test_unknown_holdout_source_fails_loudly(self):
        with self.assertRaises(ValueError):
            self.apply(self.split, holdout_source="nope")


if __name__ == "__main__":
    unittest.main()
