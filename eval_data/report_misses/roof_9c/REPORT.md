# Missed-Symbol Diagnostic Report

Model: `symbol_detector_best.pt` | dataset: `test_small_9c` | split: `test` (32 images, 283 GT instances)
Production settings: conf=0.35, imgsz=1280; tiles 653px with 65px overlap.

## Overall

| Bucket | Count | % of GT |
|---|---|---|
| Detected at prod conf 0.35 | 190 | 67.1% |
| Threshold miss (0.05-0.35, silently dropped) | 59 | 20.8% |
| Recovered only by TTA | 7 | 2.5% |
| Undetected even at 0.05 | 27 | 9.5% |

## Per-class findings (classes with misses, worst recall first)

| Class | GT | Train | R@0.35 | R@0.05 | +TTA | Diagnosis | Recommendation |
|---|---|---|---|---|---|---|---|
| HATCH DETAIL | 3 | 4 | 0.0 | 0.333 | 0.333 | threshold+data_scarcity+confused_class | found at conf 0.07-0.07 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) | only 4 train instances (<20); label more examples | often predicted as 'EXHAUST FAN' instead (2x); check label consistency between the two classes |
| HEAT DETAIL | 1 | 2 | 0.0 | 0.0 | 0.0 | data_scarcity+confused_class | only 2 train instances (<20); label more examples | often predicted as 'ROOF DRAIN' instead (1x); check label consistency between the two classes |
| PIPE PENETRATION | 34 | 58 | 0.206 | 0.618 | 0.676 | threshold+confused_class | found at conf 0.06-0.30 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) | often predicted as 'ROOF DRAIN' instead (4x); check label consistency between the two classes |
| PLUMBING STACK | 47 | 96 | 0.234 | 0.787 | 0.872 | threshold+confused_class | found at conf 0.06-0.30 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) | often predicted as 'ROOF DRAIN' instead (1x); check label consistency between the two classes |
| SKYLIGHT | 2 | 5 | 0.5 | 0.5 | 0.5 | data_scarcity+confused_class | only 5 train instances (<20); label more examples | often predicted as 'ROOF DRAIN' instead (1x); check label consistency between the two classes |
| EXHAUST FAN | 16 | 31 | 0.812 | 0.938 | 0.938 | threshold+tile_edge+confused_class | found at conf 0.18-0.35 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) | 2/3 misses within 5% of an image edge; symbol 35px vs 65px tile overlap - increase overlap | often predicted as 'ROOF DRAIN' instead (1x); check label consistency between the two classes |
| CURB | 80 | 519 | 0.863 | 0.975 | 0.975 | threshold+confused_class | found at conf 0.13-0.32 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) | often predicted as 'EXHAUST FAN' instead (1x); check label consistency between the two classes |
| ROOF DRAIN | 100 | 200 | 0.89 | 0.96 | 0.97 | threshold | found at conf 0.08-0.33 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) |

## Files

- `miss_details.csv` - every missed GT instance with status, conf, size, edge distance
- `per_class_summary.csv` - full per-class table (including clean classes)
- `overlays/` - GT (red) vs low-conf predictions (orange) for representative misses

## Legend for diagnosis tags

- **threshold**: model finds it below prod conf 0.35 - not a detection failure
- **data_scarcity**: under 20 training instances - label more, don't retune
- **tiny_p2_candidate**: below the P3 stride-8 floor at inference scale - yolov8-p2.yaml retrain
- **tile_edge**: misses cluster near image edges - increase tile overlap above symbol size
- **confused_class**: detected but as a different class - check labels for the confused pair
- **hard_examples**: invisible even at conf 0.05 - inspect overlays for appearance drift