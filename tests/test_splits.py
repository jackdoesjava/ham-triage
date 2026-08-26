import numpy as np
import pandas as pd
import pytest

from ham_triage.config import Paths
from ham_triage.splits import audit_split, image_level_split


@pytest.fixture(scope="module")
def meta():
    path = Paths().meta
    if not path.exists():
        pytest.skip("image cache not built")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def split(meta):
    return audit_split(meta, seed=0)


def lesions(meta, mask):
    return set(meta.lesion_id[mask])


def test_no_lesion_crosses_test_cal_or_clean_train(meta, split):
    test, cal, train = (lesions(meta, split[c]) for c in ("test", "cal", "clean_train"))
    assert not test & cal and not test & train and not cal & train


def test_cal_has_one_image_per_lesion(meta, split):
    assert meta.lesion_id[split.cal].is_unique


def test_leaky_train_differs_from_clean_only_by_test_siblings(meta, split):
    added = split.leaky_train & ~split.clean_train
    removed = split.clean_train & ~split.leaky_train
    test_lesions = lesions(meta, split.test)
    assert meta.lesion_id[added].isin(test_lesions).all()
    assert not meta.lesion_id[removed].isin(test_lesions).any()
    # every non-test image of a test lesion is in, nothing else touches test lesions
    assert added.sum() == (~split.test & meta.lesion_id.isin(test_lesions)).sum()
    assert split.leaky_train.sum() == split.clean_train.sum()
    assert meta.dx[split.leaky_train].value_counts().equals(meta.dx[split.clean_train].value_counts())


def test_test_set_is_the_naive_one(meta, split):
    assert np.array_equal(split.test.values, image_level_split(meta, seed=0).test.values)


def test_masks_are_disjoint_and_dx_is_constant_within_lesion(meta, split):
    assert (split[["test", "cal", "clean_train"]].sum(axis=1) <= 1).all()
    assert (meta.groupby("lesion_id").dx.nunique() == 1).all()
