"""Export the fine-tuned device classifier to ONNX for fast local inference.

STATUS: stub - requires a trained classifier checkpoint that does not exist yet.

Intended flow once a checkpoint exists (trained with timm, e.g. resnet18 or
efficientnet_b0, on symbol crops labeled with the classes in
ml/classifier.py::DEVICE_TYPES - class order must match exactly):

    import timm, torch
    model = timm.create_model("resnet18", num_classes=len(DEVICE_TYPES))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224),
        "backend/models/device_classifier.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
    )

The ONNX file is consumed by ml/classifier.py::OnnxDeviceClassifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?", help="Trained .pt checkpoint")
    args = parser.parse_args()

    if args.checkpoint is None or not args.checkpoint.exists():
        sys.exit(
            "No trained classifier checkpoint provided.\n\n"
            "This POC intentionally ships without trained weights - the pipeline\n"
            "runs on mock model outputs (HVAC_USE_MOCK_MODELS=true).\n"
            "Train a classifier on annotated symbol crops first, then run:\n"
            "  python scripts/export_model.py path/to/checkpoint.pt"
        )

    raise NotImplementedError(
        "Checkpoint found, but the export step is not implemented yet. "
        "See the module docstring for the intended torch.onnx.export recipe."
    )


if __name__ == "__main__":
    main()
