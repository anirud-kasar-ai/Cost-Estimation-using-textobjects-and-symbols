"""Build the 8-crop ground-truth set used by eval/score.py.

These 653px tiles encode the documented look-alike pairs from Garland-style
telecom bid drawings (the same conventions as Real data/). Boxes are the
exact ink bounds used when drawing, so they match CV contour geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CROPS = ROOT / "crops"
GT = ROOT / "gt"
SIZE = 653


def _font(size: int = 28) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _blank() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    return img, ImageDraw.Draw(img)


def _inst(key: str, box: tuple[int, int, int, int], qty: int = 1) -> dict:
    x1, y1, x2, y2 = box
    return {"key": key, "qty": qty, "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def crop_hash_standalone() -> tuple[Image.Image, list[dict]]:
    img, draw = _blank()
    tris = [
        [(120, 140), (100, 180), (140, 180)],
        [(320, 300), (300, 340), (340, 340)],
        [(480, 200), (460, 240), (500, 240)],
    ]
    instances = []
    for pts in tris:
        draw.polygon(pts, fill=(0, 0, 0))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        instances.append(_inst("#", (min(xs), min(ys), max(xs), max(ys))))
    return img, instances


def crop_ap_circle() -> tuple[Image.Image, list[dict]]:
    img, draw = _blank()
    draw.ellipse((270, 270, 370, 370), outline=(0, 0, 0), width=3)
    tri = [(320, 288), (300, 328), (340, 328)]
    draw.polygon(tri, fill=(0, 0, 0))
    draw.text((302, 378), "AP", fill=(0, 0, 0), font=_font(26))
    return img, [_inst("AP", (270, 270, 370, 370))]


def crop_hourglass() -> tuple[Image.Image, list[dict]]:
    img, draw = _blank()
    left = [(180, 300), (180, 340), (250, 320)]
    right = [(380, 300), (380, 340), (310, 320)]
    draw.polygon(left, fill=(0, 0, 0))
    draw.rectangle((255, 308, 295, 332), outline=(0, 0, 0), width=2)
    draw.line((255, 308, 295, 332), fill=(0, 0, 0), width=2)
    draw.line((295, 308, 255, 332), fill=(0, 0, 0), width=2)
    draw.polygon(right, fill=(0, 0, 0))
    return img, [_inst("NETWORK CAMERA", (180, 300, 380, 340))]


def crop_bowtie() -> tuple[Image.Image, list[dict]]:
    img, draw = _blank()
    draw.rectangle((276, 276, 376, 376), outline=(0, 0, 0), width=3)
    draw.polygon([(278, 278), (278, 374), (326, 326)], fill=(0, 0, 0))
    draw.polygon([(374, 278), (374, 374), (326, 326)], fill=(0, 0, 0))
    return img, [_inst("DATA POLE", (276, 276, 376, 376))]


def crop_mixed_lookalikes() -> tuple[Image.Image, list[dict]]:
    """All four high-risk classes on one tile."""
    img, draw = _blank()
    instances = []
    # standalone #
    htri = [(80, 80), (60, 120), (100, 120)]
    draw.polygon(htri, fill=(0, 0, 0))
    instances.append(_inst("#", (60, 80, 100, 120)))
    # AP
    draw.ellipse((70, 400, 150, 480), outline=(0, 0, 0), width=3)
    draw.polygon([(110, 418), (94, 452), (126, 452)], fill=(0, 0, 0))
    instances.append(_inst("AP", (70, 400, 150, 480)))
    # hourglass
    draw.polygon([(380, 90), (380, 124), (440, 107)], fill=(0, 0, 0))
    draw.rectangle((445, 98, 470, 116), outline=(0, 0, 0), width=2)
    draw.polygon([(530, 90), (530, 124), (475, 107)], fill=(0, 0, 0))
    instances.append(_inst("NETWORK CAMERA", (380, 90, 530, 124)))
    # bowtie
    draw.rectangle((430, 430, 510, 510), outline=(0, 0, 0), width=3)
    draw.polygon([(432, 432), (432, 508), (470, 470)], fill=(0, 0, 0))
    draw.polygon([(508, 432), (508, 508), (470, 470)], fill=(0, 0, 0))
    instances.append(_inst("DATA POLE", (430, 430, 510, 510)))
    return img, instances


def crop_jhook() -> tuple[Image.Image, list[dict]]:
    img, draw = _blank()
    font = _font(32)
    for x in (140, 200, 260):
        draw.text((x, 310), "J", fill=(0, 0, 0), font=font)
    return img, [_inst("J-HOOK (SINGLE/STACKED)", (140, 310, 290, 350))]


def crop_jbox() -> tuple[Image.Image, list[dict]]:
    img, draw = _blank()
    draw.text((300, 300), "J", fill=(0, 0, 0), font=_font(32))
    return img, [_inst("J", (300, 300, 330, 340))]


def crop_dense_wall() -> tuple[Image.Image, list[dict]]:
    img, draw = _blank()
    draw.line((40, 320, 620, 320), fill=(0, 0, 0), width=2)
    draw.rectangle((180, 300, 230, 340), outline=(0, 0, 0), width=2)
    draw.text((240, 302), "5400", fill=(0, 0, 0), font=_font(22))
    tri = [(400, 290), (385, 325), (415, 325)]
    draw.polygon(tri, fill=(0, 0, 0))
    draw.text((420, 298), "2", fill=(0, 0, 0), font=_font(24))
    return img, [
        _inst("SURFACE RACEWAY (WM5400)", (180, 300, 310, 340)),
        _inst("#", (385, 290, 415, 325), qty=2),
    ]


BUILDERS = {
    "01_hash_standalone": crop_hash_standalone,
    "02_ap_circle": crop_ap_circle,
    "03_hourglass_camera": crop_hourglass,
    "04_bowtie_pole": crop_bowtie,
    "05_mixed_lookalikes": crop_mixed_lookalikes,
    "06_jhook_run": crop_jhook,
    "07_junction_j": crop_jbox,
    "08_dense_wall": crop_dense_wall,
}


def main() -> None:
    CROPS.mkdir(parents=True, exist_ok=True)
    GT.mkdir(parents=True, exist_ok=True)
    index = []
    for name, fn in BUILDERS.items():
        img, instances = fn()
        png = CROPS / f"{name}.png"
        img.save(png)
        rec = {
            "id": name,
            "image": f"crops/{name}.png",
            "width": SIZE,
            "height": SIZE,
            "instances": instances,
        }
        (GT / f"{name}.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8"
        )
        index.append(rec)
    (GT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {len(index)} crops to {CROPS}")


if __name__ == "__main__":
    main()
