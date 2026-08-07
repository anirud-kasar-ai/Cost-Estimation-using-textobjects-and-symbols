"""ML pipeline modules.

Each model wrapper (ROI detector, symbol detector, classifier, OCR) exposes a
small interface with two interchangeable backends:

- a deterministic mock (default, ``HVAC_USE_MOCK_MODELS=true``) so the whole
  pipeline runs without trained weights or heavy ML dependencies, and
- a real backend (Detectron2 / PaddleOCR / ONNX Runtime) that is lazily
  imported and used once trained weights are dropped into ``backend/models/``.
"""
