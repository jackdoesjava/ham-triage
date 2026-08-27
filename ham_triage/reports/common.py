import json

import numpy as np
import pandas as pd

from ..config import CLASSES, Paths
from ..stats import ci, recall_per_class

MEL = CLASSES.index("mel")
TREAT = np.isin(np.arange(len(CLASSES)), [CLASSES.index(c) for c in ("mel", "bcc", "akiec")])
METRICS = ["acc", "bal_acc"] + list(CLASSES)


def load_logits(paths: Paths, run_id: str, name: str = "logits") -> np.ndarray:
    return pd.read_parquet(paths.results / "runs" / run_id / f"{name}.parquet").filter(like="logit_").values


def load_runs(paths: Paths, prefix: str, name: str = "logits") -> dict[int, np.ndarray]:
    # runs are named {prefix}_s{seed}; the audit replications are {cond}_a{split}_s{seed}
    # and are picked up by passing prefix="clean_a1"
    runs = {}
    for d in sorted((paths.results / "runs").glob(f"{prefix}_s*")):
        if (d / f"{name}.parquet").exists():
            info = json.loads((d / "run.json").read_text())
            runs[info["seed"]] = load_logits(paths, d.name, name)
    return runs


def load_split(paths: Paths, name: str):
    meta = pd.read_parquet(paths.meta)
    split = pd.read_parquet(paths.results / "splits" / f"{name}.parquet")
    assert (split.image_id.values == meta.image_id.values).all()
    return meta, split


def metric_vector(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    rec = recall_per_class(pred, y)
    return np.concatenate([[(pred == y).mean(), np.nanmean(rec)], rec])


def interval_row(samples: np.ndarray, point: float, per_seed: np.ndarray, **keys) -> dict:
    lo, hi = ci(samples)
    return {**keys, "point": point, "lo": lo, "hi": hi, "seed_min": per_seed.min(), "seed_max": per_seed.max()}


def dump(paths: Paths, name: str, obj: dict) -> None:
    (paths.results / "derived").mkdir(parents=True, exist_ok=True)
    with open(paths.results / "derived" / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=float)
