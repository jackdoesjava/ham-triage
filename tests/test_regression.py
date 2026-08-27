import json

import pytest

from ham_triage.analyse import SECTIONS
from ham_triage.config import Paths

# every derived artifact regenerates byte-for-byte from the committed logits; this is
# the strongest reproducibility claim the repo makes, so it is a test rather than a
# sentence in the README. About a minute.
CHECKED = ["audit", "imbalance", "calibration", "conformal", "decision", "strata", "external"]


@pytest.mark.parametrize("name", CHECKED)
def test_artifact_regenerates_identically(name):
    paths = Paths()
    path = paths.results / "derived" / f"{name}.json"
    if not paths.meta.exists() or not path.exists():
        pytest.skip("data cache or artifact missing")
    fresh = json.loads(json.dumps(SECTIONS[name](paths), default=float))
    committed = json.loads(path.read_text())
    assert fresh == committed
