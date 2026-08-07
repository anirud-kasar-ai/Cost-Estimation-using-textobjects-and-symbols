"""Train the Detectron2 HVAC device symbol detector.

STATUS: stub - training requires annotated data that does not exist yet.

Required inputs (place under ml-training/data/symbols/):
    images/                     page PNGs exported by ml/pdf_to_image.py
    annotations.json            COCO-format boxes. Either a single generic
                                "device_symbol" category (detector proposes,
                                classifier decides the sub-type - matches the
                                current pipeline design), or one category per
                                device type if you prefer a single-stage model.

Once data exists: fine-tune a COCO-pretrained Faster R-CNN, log to MLflow
(./mlruns), and write weights + config to
backend/models/symbol_detector.{pth,yaml}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "ml-training" / "data" / "symbols"


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
            "To train for real: annotate device symbol boxes with Label Studio,\n"
            "export as COCO, place the export in the path above, then implement\n"
            "the Detectron2 training loop described in this script's docstring."
        )

    raise NotImplementedError(
        "Training data found, but the training loop is not implemented yet. "
        "See the module docstring for the intended Detectron2 recipe."
    )


if __name__ == "__main__":
    main()
