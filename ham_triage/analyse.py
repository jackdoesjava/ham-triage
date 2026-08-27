import argparse

from .config import Paths
from .reports.audit import audit
from .reports.calibration import calibration
from .reports.common import dump
from .reports.conformal import conformal
from .reports.decision import decision
from .reports.external import external
from .reports.imbalance import imbalance
from .reports.strata import strata

SECTIONS = {"audit": audit, "imbalance": imbalance, "calibration": calibration, "conformal": conformal,
            "decision": decision, "strata": strata, "external": external}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sections", nargs="*", default=list(SECTIONS), help="subset of sections to rerun")
    args = p.parse_args()
    paths = Paths()
    for name in args.sections:
        print(f"--- {name} ---")
        dump(paths, name, SECTIONS[name](paths))
