import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .config import CACHE_SIZE, CLASSES, Paths, TrainConfig
from .train import build_model, predict

# ISIC 2018 challenge task 3 test set: 1512 images from the same sources as HAM10000
# plus four other centres, held out by the organisers, ground truth released in 2019.
# The organisers state it shares no lesion with HAM10000; there is no lesion_id in the
# public files, so it is bootstrapped by image.
URL = "https://isic-challenge-data.s3.amazonaws.com/2018/"
N_IMAGES = 1512


def isic_paths(paths: Paths):
    root = paths.data / "isic2018"
    return root, paths.data / "isic2018_256.npy", paths.data / "isic2018_meta.parquet"


def build_cache(paths: Paths) -> None:
    root, cache, meta_path = isic_paths(paths)
    csv = next(root.rglob("ISIC2018_Task3_Test_GroundTruth.csv"))
    gt = pd.read_csv(csv)
    onehot = gt[[c.upper() for c in CLASSES]].values
    assert (onehot.sum(axis=1) == 1).all() and len(gt) == N_IMAGES
    meta = pd.DataFrame({"image_id": gt["image"], "label": onehot.argmax(axis=1).astype("int8")})
    meta["dx"] = [CLASSES[k] for k in meta.label]
    files = {p.stem: p for p in root.rglob("*.jpg")}
    images = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=(len(meta), CACHE_SIZE, CACHE_SIZE, 3))
    for i, image_id in enumerate(tqdm(meta.image_id, desc="isic2018")):
        images[i] = np.asarray(Image.open(files[image_id]).convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC))
    images.flush()
    meta.to_parquet(meta_path, index=False)
    print(meta.dx.value_counts().to_string())


def score_all_runs(paths: Paths, device: str) -> None:
    _, cache, meta_path = isic_paths(paths)
    images, meta = np.load(cache), pd.read_parquet(meta_path)
    for run in sorted((paths.results / "runs").glob("*")):
        out = run / "isic2018.parquet"
        if out.exists() or not (run / "checkpoint.pt").exists():
            continue
        info = json.loads((run / "run.json").read_text())
        cfg = TrainConfig(**info["config"])
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location=device))
        logits = predict(model, images, cfg, device)
        frame = pd.DataFrame(logits, columns=[f"logit_{c}" for c in CLASSES])
        frame.insert(0, "image_id", meta.image_id.values)
        frame.to_parquet(out, index=False)
        print(f"{run.name}: acc {(logits.argmax(1) == meta.label.values).mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    paths = Paths()
    if not isic_paths(paths)[1].exists():
        build_cache(paths)
    score_all_runs(paths, args.device)
