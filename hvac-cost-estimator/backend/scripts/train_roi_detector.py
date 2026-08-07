"""Train the Detectron2 title-block / ROI detector.

STATUS: stub - training requires annotated data that does not exist yet.

Required inputs (place under ml-training/data/roi/):
    images/                     page PNGs exported by ml/pdf_to_image.py
    annotations.json            COCO-format boxes with a single "title_block"
                                category (export from Label Studio:
                                Project -> Export -> COCO)

Once data exists, the training loop below is the standard Detectron2 recipe:
fine-tune a COCO-pretrained Faster R-CNN (e.g. faster_rcnn_R_50_FPN_3x) with
NUM_CLASSES=1, log metrics to MLflow (file-based, ./mlruns), and write final
weights + config to backend/models/roi_detector.{pth,yaml}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "ml-training" / "data" / "roi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    annotations = args.data_dir / "annotations.json"
    if not annotations.exists():
        sys.exit(
            "No training data found.\n"
            f"  Expected COCO annotations at: {annotations}\n\n"
            "This POC intentionally ships without trained weights - the pipeline\n"
            "runs on mock model outputs (HVAC_USE_MOCK_MODELS=true).\n"
            "To train for real: annotate title-block boxes on page images with\n"
            "Label Studio (label-studio, run locally), export as COCO, place the\n"
            "export in the path above, then implement the Detectron2 training\n"
            "loop described in this script's docstring."
        )

    raise NotImplementedError(
        "Training data found, but the training loop is not implemented yet. "
        "See the module docstring for the intended Detectron2 recipe."
    )


if __name__ == "__main__":
    main()
