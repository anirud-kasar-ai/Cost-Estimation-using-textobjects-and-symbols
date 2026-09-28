"""Audit technical_symbols PDFs vs YOLO class map coverage."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pipeline.legend_match import _norm, match_class_to_legend
from pipeline.legend_ocr import LegendEntry, parse_legend_file
from pipeline.yolo_class_map import _CLASS_TO_LEGEND, resolve_class_map
from pipeline.yolo_detect import get_model

PDF_DIR = Path(
    r"d:\Cost Estimation Using Text and Object\drawing-zoom-split\storage\technical_symbols"
)


def _full_label(e: LegendEntry) -> str:
    d = (e.description or "").strip()
    det = (e.detail_ref or "").strip()
    return f"{d}, {det}" if det else d


def main() -> None:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"PDFs: {len(pdfs)}")

    try:
        model = get_model()
        yolo_classes = sorted({str(v) for v in (model.names or {}).values()})
    except Exception as exc:  # noqa: BLE001
        yolo_classes = []
        print("MODEL_LOAD_FAIL", exc)
    print("YOLO_MODEL_CLASSES:", yolo_classes)

    by_desc: dict[str, dict] = defaultdict(
        lambda: {"pdfs": set(), "details": set(), "raws": set(), "numbers": set()}
    )
    parse_errors: list[tuple[str, str]] = []
    per_pdf: dict[str, list[LegendEntry]] = {}

    for pdf in pdfs:
        try:
            entries = parse_legend_file(pdf)
        except Exception as exc:  # noqa: BLE001
            parse_errors.append((pdf.name, str(exc)))
            continue
        per_pdf[pdf.name] = entries
        print(f"\n=== {pdf.name} ({len(entries)} entries) ===")
        for e in entries:
            key = _norm(e.description)
            detail = (e.detail_ref or "").strip()
            full = _full_label(e)
            by_desc[key]["pdfs"].add(pdf.name)
            if detail:
                by_desc[key]["details"].add(detail)
            by_desc[key]["raws"].add(full)
            by_desc[key]["numbers"].add(e.number)
            print(f"  {e.number:3d}. {full}")

    print("\n\n===== UNIQUE LEGEND DESCRIPTIONS =====")
    unmatched_legends: list[tuple] = []
    matched_legends: list[tuple] = []
    class_pool = sorted(set(list(_CLASS_TO_LEGEND.keys()) + yolo_classes))

    for _desc_n, info in sorted(
        by_desc.items(), key=lambda x: min(x[1]["numbers"]) if x[1]["numbers"] else 999
    ):
        legend: list[LegendEntry] = []
        for i, raw in enumerate(sorted(info["raws"])):
            if ", DTL" in raw.upper() or ", OTL" in raw.upper():
                parts = raw.split(",", 1)
                d = parts[0].strip()
                det = parts[1].strip() if len(parts) > 1 else None
            else:
                d, det = raw, None
            legend.append(
                LegendEntry(number=i + 1, description=d, detail_ref=det, raw_line=raw)
            )

        hits = []
        for cls in class_pool:
            m = match_class_to_legend(cls, legend, min_score=0.55)
            if m:
                hits.append((cls, m.score, m.reason, m.legend_number))
        sample = sorted(info["raws"])[0]
        pdf_count = len(info["pdfs"])
        details = ", ".join(sorted(info["details"])) or "-"
        if hits:
            matched_legends.append((sample, details, pdf_count, hits))
            print(f"OK  [{pdf_count:2d} pdfs] {sample}")
            print(f"     details={details}")
            print(f"     matched_by={[h[0] for h in hits[:8]]}")
        else:
            unmatched_legends.append(
                (sample, details, pdf_count, sorted(info["pdfs"]))
            )
            print(f"MISS[{pdf_count:2d} pdfs] {sample}")
            print(f"     details={details}")

    print("\n\n===== SUMMARY =====")
    print(f"unique_descriptions={len(by_desc)}")
    print(f"matched={len(matched_legends)} unmatched={len(unmatched_legends)}")
    print(f"parse_errors={parse_errors}")

    print("\n===== UNMATCHED (no YOLO class maps to this legend) =====")
    for sample, details, pdf_count, _pdfs in unmatched_legends:
        print(f"- {sample} | details={details} | in {pdf_count} pdfs")

    # Flat legend list of every full variant
    all_entries: list[LegendEntry] = []
    n = 1
    for info in by_desc.values():
        for raw in info["raws"]:
            if ", DTL" in raw.upper() or ", OTL" in raw.upper():
                parts = raw.split(",", 1)
                d = parts[0].strip()
                det = parts[1].strip() if len(parts) > 1 else None
            else:
                d, det = raw, None
            all_entries.append(
                LegendEntry(number=n, description=d, detail_ref=det, raw_line=raw)
            )
            n += 1

    print("\n===== YOLO MODEL CLASS → LEGEND =====")
    for cls in yolo_classes or []:
        m = match_class_to_legend(cls, all_entries)
        entry = resolve_class_map(cls)
        phrases = list(entry.legend_phrases) if entry else []
        if m:
            hit = next(e for e in all_entries if e.number == m.legend_number)
            print(
                f"{cls:30s} -> {hit.raw_line or hit.description} "
                f"(score={m.score:.2f}) phrases={phrases[:5]}"
            )
        else:
            print(f"{cls:30s} -> NO LEGEND MATCH  phrases={phrases}")

    # DTL variants per core description that appear but may not be listed as phrases
    print("\n===== DETAIL VARIANTS PER DESCRIPTION =====")
    for desc_n, info in sorted(by_desc.items(), key=lambda x: x[0]):
        if not info["details"]:
            continue
        print(f"{desc_n}:")
        for d in sorted(info["details"]):
            print(f"  - {d}")

    print("\n===== ALL FULL LEGEND STRINGS =====")
    all_fulls: set[str] = set()
    for info in by_desc.values():
        all_fulls |= info["raws"]
    for s in sorted(all_fulls, key=str.upper):
        print(s)


if __name__ == "__main__":
    main()
