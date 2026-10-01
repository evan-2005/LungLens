"""
The 245 scored test films: prediction, served-heatmap descriptors and image paths.

Built from the outputs the paper's localisation analysis already produced, so
nothing is recomputed:
  fig7_work/localisation_table.csv   descriptors of the served overlay
  fig7_work/test_manifest.csv        per-class probabilities (joined on path)
  fig7_work/overlays/<key>_*          radiograph and floored heatmap per film
"""
import csv
import os
import random
from dataclasses import dataclass

from report_eval.descriptors import Descriptors, from_table_row
from report_eval.templates import Prediction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG7 = os.path.join(ROOT, "fig7_work")
TABLE = os.path.join(FIG7, "localisation_table.csv")
MANIFEST = os.path.join(FIG7, "test_manifest.csv")
OVERLAYS = os.path.join(FIG7, "overlays")
PROB_COLUMNS = {"Normal": "p_normal", "Pneumonia": "p_pneumonia",
                "Tuberculosis": "p_tb", "Covid-19": "p_covid"}


@dataclass(frozen=True)
class Film:
    key: str
    source: str
    group: str
    true_cls: str
    prediction: Prediction
    descriptors: Descriptors       # the served heatmap: the ground truth for scoring

    @property
    def orig_png(self):
        return os.path.join(OVERLAYS, f"{self.key}_orig.png")

    @property
    def mask_npy(self):
        return os.path.join(OVERLAYS, f"{self.key}_mask.npy")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_films(table=TABLE, manifest=MANIFEST):
    probs_by_path = {r["path"]: r for r in _read(manifest)}
    films = []
    for r in _read(table):
        m = probs_by_path.get(r["path"])
        if m is None:
            raise KeyError(f"{r['key']} has no probabilities in {manifest}")
        probs = {c: float(m[col]) for c, col in PROB_COLUMNS.items()}
        pred = Prediction(top_cls=r["app_pred"], top_prob=float(r["conf"]), probs=probs,
                          mask_source=r["mask_source"])
        films.append(Film(key=r["key"], source=r["source"], group=r["group"],
                          true_cls=r["true"], prediction=pred, descriptors=from_table_row(r)))
    return films


def derangement(keys, seed=42):
    """
    Map every key to a different key (Sattolo's algorithm: one random cycle).

    Used for the permuted-heatmap control H5: each film keeps its own
    prediction but receives another film's heatmap.
    """
    keys = list(keys)
    if len(keys) < 2:
        raise ValueError("A derangement needs at least two items.")
    order = keys[:]
    rng = random.Random(seed)
    for i in range(len(order) - 1, 0, -1):
        j = rng.randrange(i)
        order[i], order[j] = order[j], order[i]
    return dict(zip(keys, order))
