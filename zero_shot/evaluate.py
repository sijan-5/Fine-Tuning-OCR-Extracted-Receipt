"""
zero_shot/evaluate.py — Score zero-shot receipt extraction outputs.

Usage:
    python -m zero_shot.evaluate outputs/zero_shot_llama31_8b_paddle.jsonl \
        --detail-csv outputs/eval_llama31_8b_paddle.csv
"""

import argparse
import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def _similarity(a: str, b: str) -> float:
    """Normalized similarity [0, 1] between two strings."""
    a, b = _normalize_text(a), _normalize_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _parse_number(val) -> float | None:
    """Try to coerce a value to float. Return None on failure."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip().lstrip("$").replace(",", "")
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _normalize_date(s: str) -> str | None:
    """
    Attempt to normalize a date string to MM/DD/YYYY.
    Returns None if unparseable.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()

    # MM/DD/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"

    # MM/DD/YY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2})$", s)
    if m:
        yr = int(m.group(3))
        yr += 2000 if yr < 50 else 1900
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{yr}"

    # YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        return f"{int(m.group(2)):02d}/{int(m.group(3)):02d}/{m.group(1)}"

    # DD-Mon-YYYY or DD Mon YYYY (e.g., "25-Apr-2015")
    m = re.match(r"^(\d{1,2})[\s-]([A-Za-z]{3})[\s-](\d{4})$", s)
    if m:
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        mon = months.get(m.group(2).lower()[:3])
        if mon:
            return f"{mon:02d}/{int(m.group(1)):02d}/{m.group(3)}"

    return None


# ---------------------------------------------------------------------------
# Field-level scorers  (each returns float in [0, 1])
# ---------------------------------------------------------------------------

def score_store_name(gt, pred) -> float:
    if gt is None and pred is None:
        return 1.0
    if gt is None or pred is None:
        return 0.0
    return _similarity(str(gt), str(pred))


def score_date(gt, pred) -> float:
    gt_n = _normalize_date(str(gt)) if gt is not None else None
    pred_n = _normalize_date(str(pred)) if pred is not None else None
    if gt_n is None and pred_n is None:
        return 1.0
    if gt_n is None or pred_n is None:
        return 0.0
    return 1.0 if gt_n == pred_n else 0.0


def score_numeric(gt, pred) -> float:
    gt_n = _parse_number(gt)
    pred_n = _parse_number(pred)
    if gt_n is None and pred_n is None:
        return 1.0
    if gt_n is None or pred_n is None:
        return 0.0
    return 1.0 if abs(gt_n - pred_n) <= 0.01 else 0.0


def score_category(gt, pred) -> float:
    if gt is None and pred is None:
        return 1.0
    if gt is None or pred is None:
        return 0.0
    g = _normalize_text(str(gt))
    p = _normalize_text(str(pred))
    if g == p:
        return 1.0
    # Partial credit: substring match (e.g., "coffee" vs "coffee and drinks")
    if g in p or p in g:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Item-level scoring
# ---------------------------------------------------------------------------

def score_items(gt_items: list, pred_items: list) -> dict:
    """
    Match predicted items to GT items by name similarity (greedy, threshold 0.5).
    Returns precision, recall, f1, and mean detail score for matched pairs.
    """
    if not gt_items and not pred_items:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "detail": 1.0,
                "matched": 0, "gt_count": 0, "pred_count": 0}
    if not gt_items:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "detail": 0.0,
                "matched": 0, "gt_count": 0, "pred_count": len(pred_items)}
    if not pred_items:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "detail": 0.0,
                "matched": 0, "gt_count": len(gt_items), "pred_count": 0}

    n_gt = len(gt_items)
    n_pred = len(pred_items)

    # Build similarity matrix
    sim_matrix = []
    for gi in gt_items:
        row = []
        for pj in pred_items:
            s = _similarity(
                str(gi.get("item_name", "")),
                str(pj.get("item_name", "")),
            )
            row.append(s)
        sim_matrix.append(row)

    # Greedy matching: highest similarity first, each item used at most once
    matched_gt = set()
    matched_pred = set()
    pairs = []
    all_pairs = sorted(
        [(sim_matrix[i][j], i, j) for i in range(n_gt) for j in range(n_pred)],
        reverse=True,
    )
    for sim, i, j in all_pairs:
        if sim < 0.5:
            break
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        pairs.append((i, j, sim))

    n_matched = len(pairs)
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Detail score for matched pairs
    detail_scores = []
    for i, j, name_sim in pairs:
        gi = gt_items[i]
        pj = pred_items[j]
        fields = [
            name_sim,
            score_numeric(gi.get("quantity"), pj.get("quantity")),
            score_numeric(gi.get("unit_price"), pj.get("unit_price")),
            score_numeric(gi.get("total_price"), pj.get("total_price")),
            score_category(gi.get("category"), pj.get("category")),
        ]
        detail_scores.append(sum(fields) / len(fields))

    mean_detail = sum(detail_scores) / len(detail_scores) if detail_scores else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detail": mean_detail,
        "matched": n_matched,
        "gt_count": n_gt,
        "pred_count": n_pred,
    }


# ---------------------------------------------------------------------------
# Row-level scoring
# ---------------------------------------------------------------------------

NUMERIC_FIELDS = ["subtotal", "tax", "total", "discount", "tip", "service_charge"]


def score_row(gt_raw, pred: dict | None) -> dict:
    """Score a single row. Returns dict of field -> score in [0, 1]."""
    # Parse ground truth (may be JSON string or already a dict)
    if isinstance(gt_raw, str):
        try:
            gt = json.loads(gt_raw)
        except (json.JSONDecodeError, TypeError):
            gt = {}
    elif isinstance(gt_raw, dict):
        gt = gt_raw
    else:
        gt = {}

    if pred is None:
        scores = {"store_name": 0.0, "date": 0.0}
        for f in NUMERIC_FIELDS:
            scores[f] = 0.0
        scores["items_f1"] = 0.0
        scores["items_detail"] = 0.0
        scores["items_precision"] = 0.0
        scores["items_recall"] = 0.0
        return scores

    scores = {}
    scores["store_name"] = score_store_name(gt.get("store_name"), pred.get("store_name"))
    scores["date"] = score_date(gt.get("date"), pred.get("date"))
    for f in NUMERIC_FIELDS:
        scores[f] = score_numeric(gt.get(f), pred.get(f))

    gt_items = gt.get("items", []) or []
    pred_items = pred.get("items", []) or []
    item_scores = score_items(gt_items, pred_items)
    scores["items_f1"] = item_scores["f1"]
    scores["items_detail"] = item_scores["detail"]
    scores["items_precision"] = item_scores["precision"]
    scores["items_recall"] = item_scores["recall"]

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot receipt extraction.")
    parser.add_argument("jsonl", help="Path to output JSONL from run_text_only.py")
    parser.add_argument("--detail-csv", default=None,
                        help="Optional path for per-row breakdown CSV")
    args = parser.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("No rows found.", file=sys.stderr)
        sys.exit(1)

    total = len(rows)
    valid_json = sum(1 for r in rows if r.get("prediction") is not None)
    empty_ocr = sum(1 for r in rows if r.get("error") == "empty OCR")
    parse_fail = total - valid_json - empty_ocr

    # Score all rows (excluding empty-OCR rows from aggregates)
    all_scores = []
    detail_rows = []
    for r in rows:
        scores = score_row(r.get("ground_truth"), r.get("prediction"))
        is_empty_ocr = r.get("error") == "empty OCR"
        detail_rows.append({
            "image_id": r.get("image_id"),
            "error": r.get("error", ""),
            **scores,
        })
        if not is_empty_ocr:
            all_scores.append(scores)

    scored_count = len(all_scores)

    # Aggregate means
    score_keys = ["store_name", "date"] + NUMERIC_FIELDS + [
        "items_f1", "items_detail", "items_precision", "items_recall"
    ]
    means = {}
    for k in score_keys:
        vals = [s[k] for s in all_scores]
        means[k] = sum(vals) / len(vals) if vals else 0.0

    # Overall: simple mean of 8 header fields + items_f1
    header_fields = ["store_name", "date"] + NUMERIC_FIELDS
    overall = (sum(means[k] for k in header_fields) + means["items_f1"]) / (len(header_fields) + 1)

    # Print summary
    print("=" * 60)
    print(f"  Evaluation: {path.name}")
    print("=" * 60)
    print(f"  Total rows:       {total}")
    print(f"  Valid JSON:       {valid_json} ({100*valid_json/total:.1f}%)")
    print(f"  Empty OCR:        {empty_ocr}  (excluded from scoring)")
    print(f"  Parse failures:   {parse_fail}")
    print(f"  Scored rows:      {scored_count}")
    print("-" * 60)
    print("  Header field accuracy (mean over all rows):")
    print(f"    store_name:       {means['store_name']:.4f}")
    print(f"    date:             {means['date']:.4f}")
    for f in NUMERIC_FIELDS:
        print(f"    {f:<18s} {means[f]:.4f}")
    print("-" * 60)
    print("  Items:")
    print(f"    precision:        {means['items_precision']:.4f}")
    print(f"    recall:           {means['items_recall']:.4f}")
    print(f"    F1:               {means['items_f1']:.4f}")
    print(f"    detail (matched): {means['items_detail']:.4f}")
    print("-" * 60)
    print(f"  OVERALL SCORE:      {overall:.4f}")
    print("=" * 60)

    # Write detail CSV
    if args.detail_csv:
        csv_path = Path(args.detail_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image_id", "error"] + score_keys
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(detail_rows)
        print(f"\nPer-row breakdown written to {csv_path}")


if __name__ == "__main__":
    main()
