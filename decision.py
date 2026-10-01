"""
Calibrated, pneumonia-sensitive decision rule (improvement #10).

Two adjustments on top of the classifier's logits, both fitted on the
validation split only (calibration/fit_decision.py) and saved to
decision_config.json with the SHA-256 of the checkpoint they were fitted for:

  temperature          logits / T before softmax, so displayed confidences
                       match observed accuracy. Never changes the argmax.
  pneumonia_threshold  if the top class is Normal but P(pneumonia) >= t, report
                       Pneumonia. A missed treatable infection costs more than
                       a false alarm, and pneumonia-predicted-Normal is the
                       dominant error. It never overrides TB or Covid-19.

The app applies the config only when the hash matches the checkpoint it serves,
so a config can never silently apply to a different model.
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

NORMAL_IDX, PNEUMONIA_IDX = 0, 1
CONFIG_PATH = "decision_config.json"


@dataclass(frozen=True)
class DecisionConfig:
    temperature: float = 1.0
    pneumonia_threshold: Optional[float] = None
    checkpoint_sha256: str = ""


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def decide(logits, cfg):
    """(probabilities, predicted index) for one image's logits."""
    probs = softmax(logits, cfg.temperature)
    idx = int(np.argmax(probs))
    if (cfg.pneumonia_threshold is not None and idx == NORMAL_IDX
            and probs[PNEUMONIA_IDX] >= cfg.pneumonia_threshold):
        idx = PNEUMONIA_IDX
    return probs, idx


def decide_batch(logits, cfg):
    return np.array([decide(row, cfg)[1] for row in np.asarray(logits)])


def fit_temperature(logits, labels, grid=np.linspace(0.25, 8.0, 311)):
    """Temperature minimising validation negative log-likelihood (grid search)."""
    logits = np.asarray(logits, np.float64)
    labels = np.asarray(labels)
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = softmax(logits, t)[np.arange(len(labels)), labels]
        nll = -np.mean(np.log(np.clip(p, 1e-12, None)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def save_config(cfg, path=CONFIG_PATH, extra=None):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({**asdict(cfg), **(extra or {})}, fh, indent=2)


def load_config_for(checkpoint_path, path=CONFIG_PATH):
    """The saved config if it was fitted for this exact checkpoint, else (default, reason)."""
    if not os.path.exists(path):
        return DecisionConfig(), "no decision_config.json; using plain softmax"
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    cfg = DecisionConfig(temperature=float(raw.get("temperature", 1.0)),
                         pneumonia_threshold=raw.get("pneumonia_threshold"),
                         checkpoint_sha256=raw.get("checkpoint_sha256", ""))
    if not os.path.exists(checkpoint_path) or sha256_of(checkpoint_path) != cfg.checkpoint_sha256:
        return DecisionConfig(), ("decision_config.json was fitted for a different checkpoint; "
                                  "ignoring it and using plain softmax")
    return cfg, "applied decision_config.json"
