import numpy as np
import pandas as pd

from .config import Paths


def image_level_split(meta: pd.DataFrame, seed: int, test_frac: float = 0.2) -> pd.DataFrame:
    # what most HAM10000 papers do: shuffle images and cut, ignoring lesion_id
    rng = np.random.default_rng(seed)
    test = np.zeros(len(meta), dtype=bool)
    test[rng.permutation(len(meta))[: round(test_frac * len(meta))]] = True
    train_lesions = set(meta.lesion_id[~test])
    return pd.DataFrame({
        "image_id": meta.image_id,
        "train": ~test,
        "test": test,
        "sibling_in_train": test & meta.lesion_id.isin(train_lesions).values,
    })


if __name__ == "__main__":
    paths = Paths()
    meta = pd.read_parquet(paths.meta)
    split = image_level_split(meta, seed=0)
    (paths.results / "splits").mkdir(parents=True, exist_ok=True)
    split.to_parquet(paths.results / "splits" / "image_level.parquet", index=False)
    leaked = split.sibling_in_train[split.test]
    print(f"train {split.train.sum()}  test {split.test.sum()}  "
          f"test images with a training sibling: {leaked.mean():.1%}")
    print(leaked.groupby(meta.dx[split.test]).mean().round(3).to_string())
