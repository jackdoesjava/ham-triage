from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# index order used everywhere: labels, logits columns, cost matrix rows
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

# what the Kaggle copy of HAM10000 is supposed to contain; the loader asserts these
# so a silently different dataset version fails at load time rather than in a table
N_IMAGES = 10015
N_LESIONS = 7470
CLASS_COUNTS = {"nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514, "akiec": 327, "vasc": 142, "df": 115}


@dataclass(frozen=True)
class Paths:
    data: Path = REPO / "data"
    results: Path = REPO / "results"

    @property
    def images(self) -> Path:
        return self.data / "images_256.npy"

    @property
    def meta(self) -> Path:
        return self.data / "meta.parquet"


CACHE_SIZE = 256  # cached square side; train crops 224 out of this, eval centre-crops
