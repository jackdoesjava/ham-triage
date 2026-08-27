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


def one_image_per_lesion(mask: np.ndarray, lesion: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = rng.permutation(np.flatnonzero(mask))
    out = np.zeros(len(mask), dtype=bool)
    out[pd.Series(shuffled).groupby(lesion[shuffled]).first().values] = True
    return out


def audit_split(meta: pd.DataFrame, seed: int, test_frac: float = 0.2, cal_frac: float = 0.15) -> pd.DataFrame:
    """Paired leakage design: one test set, two training sets that differ only in
    whether images of the test lesions are in them.

    test          the image-level test set from image_level_split, kept as is, so the
                  leaky condition is exactly what a random split does
    cal           15% of the lesions that have no image in test, ONE image per lesion,
                  so calibration points are exchangeable units; the other images of
                  those lesions are used nowhere
    clean_train   everything else that shares no lesion with test or cal
    leaky_train   clean_train with a class-matched random subset swapped out for the
                  sibling images of the test lesions; same size, same class histogram

    Bootstrap by lesion_id on test; siblings inside test are not independent. This
    split is the audit instrument only: its test set is drawn by image, which
    over-samples multi-image lesions, so nothing downstream should use it.
    """
    naive = image_level_split(meta, seed, test_frac)
    test = naive.test.values
    lesion = meta.lesion_id.values
    label = meta.label.values
    sibling = ~test & np.isin(lesion, lesion[test])
    pool = ~test & ~sibling

    rng = np.random.default_rng([seed, 1])
    pool_lesions = np.unique(lesion[pool])
    cal_lesions = rng.choice(pool_lesions, round(cal_frac * len(pool_lesions)), replace=False)
    in_cal_lesion = pool & np.isin(lesion, cal_lesions)
    cal = one_image_per_lesion(in_cal_lesion, lesion, rng)
    clean_train = pool & ~in_cal_lesion

    leaky_train = clean_train.copy()
    for k in np.unique(label[sibling]):
        candidates = np.flatnonzero(clean_train & (label == k))
        drop = rng.choice(candidates, (sibling & (label == k)).sum(), replace=False)
        leaky_train[drop] = False
    leaky_train |= sibling
    assert leaky_train.sum() == clean_train.sum()

    return pd.DataFrame({
        "image_id": meta.image_id,
        "test": test,
        "cal": cal,
        "clean_train": clean_train,
        "leaky_train": leaky_train,
        "sibling_in_leaky_train": test & np.isin(lesion, lesion[leaky_train]),
    })


def lesion_level_split(meta: pd.DataFrame, seed: int, test_frac: float = 0.2, cal_frac: float = 0.15) -> pd.DataFrame:
    """The split everything downstream of the audit uses. Both cal and test are drawn
    by lesion, so a calibration point and a test lesion are exchangeable and the
    conformal guarantee applies as stated; cal keeps one image per lesion, test keeps
    all images of its lesions and is bootstrapped by lesion. Train is everything else,
    about 6900 images, which is the deployable model rather than the audit's
    size-matched one."""
    rng = np.random.default_rng([seed, 2])
    lesion = meta.lesion_id.values
    order = rng.permutation(np.unique(lesion))
    n_test = round(test_frac * len(order))
    test_lesions, rest = order[:n_test], order[n_test:]
    cal_lesions = rest[: round(cal_frac * len(rest))]
    test = np.isin(lesion, test_lesions)
    in_cal_lesion = np.isin(lesion, cal_lesions)
    return pd.DataFrame({
        "image_id": meta.image_id,
        "train": ~test & ~in_cal_lesion,
        "cal": one_image_per_lesion(in_cal_lesion, lesion, rng),
        "test": test,
    })


if __name__ == "__main__":
    paths = Paths()
    meta = pd.read_parquet(paths.meta)
    (paths.results / "splits").mkdir(parents=True, exist_ok=True)
    image_level_split(meta, seed=0).to_parquet(paths.results / "splits" / "image_level.parquet", index=False)

    for seed in (0, 1, 2):
        split = audit_split(meta, seed=seed)
        name = "audit" if seed == 0 else f"audit_s{seed}"
        split.to_parquet(paths.results / "splits" / f"{name}.parquet", index=False)
        leaked = split.sibling_in_leaky_train[split.test]
        print(f"{name}: test {split.test.sum()} cal {split.cal.sum()} train {split.clean_train.sum()}  "
              f"leaked {leaked.mean():.1%} (mel {leaked[meta.dx[split.test] == 'mel'].mean():.1%})")

    split = lesion_level_split(meta, seed=0)
    split.to_parquet(paths.results / "splits" / "lesion.parquet", index=False)
    for col in ("train", "cal", "test"):
        m = split[col].values
        print(f"lesion/{col:5s} {m.sum():5d} images  {meta.lesion_id[m].nunique():5d} lesions   "
              + "  ".join(f"{c}:{n}" for c, n in meta.dx[m].value_counts().items()))
