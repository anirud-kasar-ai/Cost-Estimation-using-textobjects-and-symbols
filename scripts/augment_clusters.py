"""Generate cluster-augmented training data for the roof detector.

The live failure mode is clusters of 2-3 touching symbols that the model was
never trained on. This script builds a new dataset:

  eval_data/train_small_clusters/
    train/images,labels   original 62 images + N augmented copies where real
                          GT symbol crops are pasted as touching clusters
    valid -> reuses eval_data/test_small_9c/valid (untouched, no synthetics)

Synthetic rule: pick 2-3 GT crops (same class 70% of the time - the real
failure), place them edge-to-edge with -2..+2 px gap on an empty area of a
train tile, append YOLO labels for the pasted boxes.

Usage (from repo root):  python scripts/augment_clusters.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "eval_data" / "train_small" / "train"
VAL = ROOT / "eval_data" / "test_small_9c" / "valid"
OUT = ROOT / "eval_data" / "train_small_clusters"
NAMES = [
    "CURB", "EXHAUST FAN", "HATCH DETAIL", "HEAT DETAIL", "PIPE HOUSING",
    "PIPE PENETRATION", "PLUMBING STACK", "ROOF DRAIN", "SKYLIGHT",
]
VARIANTS_PER_TILE = 2  # augmented copies per source tile
CLUSTERS_PER_IMAGE = (1, 3)
SAME_CLASS_P = 0.70
MAX_PLACE_TRIES = 60
SEED = 20260923


def read_labels(path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            cid = int(p[0])
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append((cid, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def write_labels(path: Path, boxes, w: int, h: int) -> None:
    lines = []
    for cid, x1, y1, x2, y2 in boxes:
        # negative gaps can push a pasted crop past the image edge; clamp
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        lines.append(
            f"{cid} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")


def overlaps(box, others, pad: float = 6.0) -> bool:
    x1, y1, x2, y2 = box
    for _, ox1, oy1, ox2, oy2 in others:
        if not (x2 + pad < ox1 or ox2 + pad < x1 or y2 + pad < oy1 or oy2 + pad < y1):
            return True
    return False


def main() -> None:
    rng = random.Random(SEED)
    img_out = OUT / "train" / "images"
    lbl_out = OUT / "train" / "labels"
    for d in (img_out, lbl_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # --- collect symbol crops from all train images ---
    crops: dict[int, list[Image.Image]] = {}
    tiles: list[tuple[Path, list]] = []
    for img_path in sorted((SRC / "images").glob("*")):
        img = Image.open(img_path).convert("RGB")
        boxes = read_labels(SRC / "labels" / (img_path.stem + ".txt"), img.width, img.height)
        tiles.append((img_path, boxes))
        for cid, x1, y1, x2, y2 in boxes:
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crops.setdefault(cid, []).append(img.crop((int(x1), int(y1), int(x2), int(y2))))

    n_crops = sum(len(v) for v in crops.values())
    print(f"source: {len(tiles)} images, {n_crops} usable symbol crops "
          f"({ {NAMES[k]: len(v) for k, v in sorted(crops.items())} })")

    # --- copy originals ---
    for img_path, _ in tiles:
        shutil.copy2(img_path, img_out / img_path.name)
        lbl = SRC / "labels" / (img_path.stem + ".txt")
        if lbl.is_file():
            shutil.copy2(lbl, lbl_out / lbl.name)

    # --- synthesize cluster variants (tiles only, skip huge full-wing images) ---
    made = pasted = 0
    for img_path, boxes in tiles:
        img0 = Image.open(img_path).convert("RGB")
        if max(img0.size) > 1200:
            continue
        for v in range(VARIANTS_PER_TILE):
            img = img0.copy()
            new_boxes = list(boxes)
            n_clusters = rng.randint(*CLUSTERS_PER_IMAGE)
            added_any = False
            for _ in range(n_clusters):
                size = rng.choice([2, 2, 3])  # mostly pairs
                if rng.random() < SAME_CLASS_P:
                    cid = rng.choice([c for c, v_ in crops.items() if len(v_) >= 2])
                    members = [(cid, rng.choice(crops[cid])) for _ in range(size)]
                else:
                    members = [
                        (cid, rng.choice(crops[cid]))
                        for cid in rng.sample(list(crops.keys()), min(size, len(crops)))
                    ]
                cw = sum(c.width for _, c in members) + 4 * len(members)
                ch = max(c.height for _, c in members) + 4
                placed = None
                for _ in range(MAX_PLACE_TRIES):
                    x = rng.randint(0, max(0, img.width - cw))
                    y = rng.randint(0, max(0, img.height - ch))
                    if not overlaps((x, y, x + cw, y + ch), new_boxes):
                        placed = (x, y)
                        break
                if placed is None:
                    continue
                x, y = placed
                horizontal = rng.random() < 0.7
                for cid, crop in members:
                    gap = rng.randint(-2, 2)
                    img.paste(crop, (int(x), int(y)))
                    new_boxes.append((cid, x, y, x + crop.width, y + crop.height))
                    if horizontal:
                        x += crop.width + gap
                    else:
                        y += crop.height + gap
                    pasted += 1
                added_any = True
            if not added_any:
                continue
            stem = f"{img_path.stem}_clu{v}"
            img.save(img_out / f"{stem}.jpg", quality=95)
            write_labels(lbl_out / f"{stem}.txt", new_boxes, img.width, img.height)
            made += 1

    # --- data.yaml (val = untouched real validation set) ---
    (OUT / "data.yaml").write_text(
        f"train: {(OUT / 'train' / 'images').as_posix()}\n"
        f"val: {(VAL / 'images').as_posix()}\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )
    total = len(list(img_out.glob("*")))
    print(f"augmented: {made} synthetic images ({pasted} pasted symbols) -> "
          f"{total} train images total")
    print(f"dataset yaml: {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
