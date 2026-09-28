"""Build Colab zips for the roof detector.

roof_package.zip       : ONE zip with training + testing data and the model
                         (use with retrain_roof_model_colab.ipynb)
roof_training_data.zip : train + valid sets, starting weights, train script
roof_testing_data.zip  : held-out test set + data.yaml for evaluation

Usage (from repo root):  python scripts/make_colab_zips.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "colab_package"

NAMES = (
    "['CURB', 'EXHAUST FAN', 'HATCH DETAIL', 'HEAT DETAIL', 'PIPE HOUSING', "
    "'PIPE PENETRATION', 'PLUMBING STACK', 'ROOF DRAIN', 'SKYLIGHT']"
)
TRAIN_YAML = f"path: .\ntrain: train/images\nval: valid/images\nnc: 9\nnames: {NAMES}\n"
FULL_YAML = (
    f"path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n"
    f"nc: 9\nnames: {NAMES}\n"
)
TEST_YAML = (
    f"path: .\ntrain: test/images\nval: test/images\ntest: test/images\n"
    f"nc: 9\nnames: {NAMES}\n"
)


def add_tree(z: zipfile.ZipFile, folder: Path, arc_prefix: str) -> None:
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.name != "labels.cache":
            z.write(f, f"{arc_prefix}/{f.relative_to(folder).as_posix()}")


def main() -> None:
    # --- combined package: everything in one zip ---
    package_zip = ROOT / "roof_package.zip"
    with zipfile.ZipFile(package_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for split in ("train", "valid", "test"):
            add_tree(z, SRC / "data" / split, f"roof_package/data/{split}")
        z.write(SRC / "symbol_detector_best.pt", "roof_package/symbol_detector_best.pt")
        z.writestr("roof_package/data/data.yaml", FULL_YAML)
    print(package_zip, f"{package_zip.stat().st_size / 1e6:.1f} MB")

    train_zip = ROOT / "roof_training_data.zip"
    with zipfile.ZipFile(train_zip, "w", zipfile.ZIP_DEFLATED) as z:
        add_tree(z, SRC / "data" / "train", "roof_training/data/train")
        add_tree(z, SRC / "data" / "valid", "roof_training/data/valid")
        z.write(SRC / "symbol_detector_best.pt", "roof_training/symbol_detector_best.pt")
        z.write(SRC / "train_colab.py", "roof_training/train_colab.py")
        z.writestr("roof_training/data/data.yaml", TRAIN_YAML)

    test_zip = ROOT / "roof_testing_data.zip"
    with zipfile.ZipFile(test_zip, "w", zipfile.ZIP_DEFLATED) as z:
        add_tree(z, SRC / "data" / "test", "roof_testing/test")
        z.writestr("roof_testing/data.yaml", TEST_YAML)

    for p in (train_zip, test_zip):
        print(p, f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
