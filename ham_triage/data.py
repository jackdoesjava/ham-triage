from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import CACHE_SIZE, CLASS_COUNTS, CLASSES, N_IMAGES, N_LESIONS, Paths


def dataset_root() -> Path:
    return Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))


def load_metadata(root: Path) -> pd.DataFrame:
    csvs = list(root.rglob("HAM10000_metadata.csv"))
    assert len(csvs) == 1, csvs
    meta = pd.read_csv(csvs[0])

    # the Kaggle zip has been seen with HAM10000_images_part_1 and a lowercase copy of it
    # side by side, so glob everything and key on the image id rather than trusting dirs
    paths = {}
    for p in root.rglob("*.jpg"):
        prev = paths.setdefault(p.stem, p)
        assert prev.stat().st_size == p.stat().st_size, (prev, p)
    meta["path"] = meta["image_id"].map(paths)
    assert meta["path"].notna().all(), meta.loc[meta["path"].isna(), "image_id"].tolist()[:5]

    assert len(meta) == N_IMAGES
    assert meta["lesion_id"].nunique() == N_LESIONS
    assert meta["dx"].value_counts().to_dict() == CLASS_COUNTS
    meta["label"] = meta["dx"].map(CLASSES.index).astype("int8")
    return meta.sort_values("image_id").reset_index(drop=True)


def build_cache(meta: pd.DataFrame, paths: Paths) -> None:
    paths.data.mkdir(exist_ok=True)
    shape = (len(meta), CACHE_SIZE, CACHE_SIZE, 3)
    images = np.lib.format.open_memmap(paths.images, mode="w+", dtype=np.uint8, shape=shape)
    for i, p in enumerate(tqdm(meta["path"], desc="decoding")):
        # squash 600x450 to a square instead of cropping: the distortion is identical for
        # every image and every split so it cannot move a comparison, and it keeps the
        # lesion border in frame, which a centre crop would throw away
        img = Image.open(p).convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC)
        images[i] = np.asarray(img)
    images.flush()
    meta.drop(columns="path").to_parquet(paths.meta, index=False)


def load_cache(paths: Paths = Paths()) -> tuple[np.ndarray, pd.DataFrame]:
    meta = pd.read_parquet(paths.meta)
    images = np.load(paths.images)  # ~2 GB, fits in RAM, no point memmapping it
    assert len(images) == len(meta)
    return images, meta


if __name__ == "__main__":
    meta = load_metadata(dataset_root())
    build_cache(meta, Paths())
    print(meta["dx"].value_counts().to_string())
