# Missed-Symbol Diagnostic Report

Model: `symbol_detector_tech.pt` | dataset: `test_project` | split: `test` (256 images, 1176 GT instances)
Production settings: conf=0.35, imgsz=1280; tiles 653px with 65px overlap.

## Overall

| Bucket | Count | % of GT |
|---|---|---|
| Detected at prod conf 0.35 | 1090 | 92.7% |
| Threshold miss (0.05-0.35, silently dropped) | 31 | 2.6% |
| Recovered only by TTA | 11 | 0.9% |
| Undetected even at 0.05 | 44 | 3.7% |

## Per-class findings (classes with misses, worst recall first)

| Class | GT | Train | R@0.35 | R@0.05 | +TTA | Diagnosis | Recommendation |
|---|---|---|---|---|---|---|---|
| classroom 15 | 1 | 1 | 0.0 | 0.0 | 0.0 | data_scarcity | only 1 train instances (<20); label more examples |
| object | 1 | 1 | 0.0 | 0.0 | 0.0 | data_scarcity+tile_edge+confused_class | only 1 train instances (<20); label more examples | 1/1 misses within 5% of an image edge; symbol 70px vs 65px tile overlap - increase overlap | often predicted as 'SURFACE REACEWAY WM 5400' instead (1x); check label consistency between the two classes |
| J HOOK | 5 | 365 | 0.2 | 0.6 | 0.6 | threshold+tile_edge | found at conf 0.15-0.32 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) | 2/4 misses within 5% of an image edge; symbol 238px vs 65px tile overlap - increase overlap |
| DATA PERMANENT LINK | 105 | 368 | 0.733 | 0.743 | 0.752 | tile_edge | 18/28 misses within 5% of an image edge; symbol 47px vs 65px tile overlap - increase overlap |
| SURFACE RACEWAY WM2300 | 100 | 241 | 0.86 | 0.95 | 0.97 | threshold+confused_class | found at conf 0.05-0.33 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) | often predicted as 'SURFACE REACEWAY WM 5400' instead (1x); check label consistency between the two classes |
| SURFACE REACEWAY WM 5400 | 177 | 602 | 0.876 | 0.955 | 0.994 | threshold+tile_edge | found at conf 0.05-0.32 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) | 11/22 misses within 5% of an image edge; symbol 63px vs 65px tile overlap - increase overlap |
| HEAT STACK | 23 | 30 | 0.913 | 0.957 | 0.957 | threshold | found at conf 0.09-0.09 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) |
| ROOF DRAIN | 102 | 102 | 0.931 | 0.961 | 0.971 | threshold | found at conf 0.17-0.35 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) |
| INTERIOR WIRELESS ACESS POINT | 39 | 116 | 0.949 | 0.949 | 0.949 | tile_edge | 1/2 misses within 5% of an image edge; symbol 79px vs 65px tile overlap - increase overlap |
| PLUMBING STACK | 39 | 40 | 0.949 | 0.949 | 0.949 | hard_examples | not found even at conf 0.05 and not tiny/rare; inspect overlays - likely appearance drift vs training data |
| CONDUIT STUB | 69 | 190 | 0.971 | 0.986 | 0.986 | threshold+tile_edge | found at conf 0.25-0.25 but dropped at 0.35; more training examples to push conf up (or lower conf for this class) | 1/2 misses within 5% of an image edge; symbol 58px vs 65px tile overlap - increase overlap |
| MODULE COMBO BOX | 72 | 193 | 0.986 | 0.986 | 0.986 | hard_examples | not found even at conf 0.05 and not tiny/rare; inspect overlays - likely appearance drift vs training data |

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