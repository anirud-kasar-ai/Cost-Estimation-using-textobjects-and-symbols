"""Second-stage verifier: low-conf detections re-judged, confident ones trusted."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline import config  # noqa: E402
from pipeline import verify  # noqa: E402
from pipeline.yolo_detect import YoloHit  # noqa: E402


def _hit(conf: float, x1=10, y1=10, x2=40, y2=40, name="CAM") -> YoloHit:
    return YoloHit(
        class_id=0,
        class_name=name,
        confidence=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_name="t.jpg",
    )


class _FakeVerifier:
    """P(false_symbol) fixed per instance; records how many crops it saw."""

    names = {0: "false_symbol", 1: "true_symbol"}

    def __init__(self, p_false: float):
        self._p = p_false
        self.calls = 0

    def predict(self, source, imgsz, verbose):
        self.calls += 1
        return [SimpleNamespace(probs=SimpleNamespace(data=[self._p, 1 - self._p]))]


def _img() -> Image.Image:
    return Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))


def test_low_conf_junk_is_dropped(monkeypatch):
    model = _FakeVerifier(p_false=0.95)
    monkeypatch.setattr(verify, "verifier_available", lambda: True)
    monkeypatch.setattr(verify, "_get_model", lambda: model)
    monkeypatch.setattr(config, "VERIFY_MAX_CONF", 0.60)
    monkeypatch.setattr(config, "VERIFY_DROP_PROB", 0.80)

    kept, dropped = verify.verify_hits([_hit(0.30)], _img())
    assert kept == []
    assert len(dropped) == 1
    assert model.calls == 1


def test_low_conf_real_symbol_is_kept(monkeypatch):
    model = _FakeVerifier(p_false=0.10)
    monkeypatch.setattr(verify, "verifier_available", lambda: True)
    monkeypatch.setattr(verify, "_get_model", lambda: model)
    monkeypatch.setattr(config, "VERIFY_MAX_CONF", 0.60)
    monkeypatch.setattr(config, "VERIFY_DROP_PROB", 0.80)

    kept, dropped = verify.verify_hits([_hit(0.30)], _img())
    assert len(kept) == 1
    assert dropped == []


def test_confident_hits_are_never_second_guessed(monkeypatch):
    model = _FakeVerifier(p_false=0.99)  # would reject anything it sees
    monkeypatch.setattr(verify, "verifier_available", lambda: True)
    monkeypatch.setattr(verify, "_get_model", lambda: model)
    monkeypatch.setattr(config, "VERIFY_MAX_CONF", 0.60)
    monkeypatch.setattr(config, "VERIFY_DROP_PROB", 0.80)

    kept, dropped = verify.verify_hits([_hit(0.85)], _img())
    assert len(kept) == 1
    assert dropped == []
    assert model.calls == 0  # never even cropped


def test_noop_when_verifier_missing(monkeypatch):
    monkeypatch.setattr(verify, "verifier_available", lambda: False)
    hits = [_hit(0.30), _hit(0.90)]
    kept, dropped = verify.verify_hits(hits, _img())
    assert kept == hits
    assert dropped == []
