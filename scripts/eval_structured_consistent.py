
import ast
import json
import re
from pathlib import Path
import argparse
from typing import Any, Dict, List, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _try_parse(text: str):
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(text)
        except Exception:
            pass
    return None


def extract_json_like(x: Any):
    """
    Robustly parse:
    - dict / list (already parsed)
    - json string
    - markdown fenced json
    - bare {...} / [...]
    """
    if isinstance(x, (dict, list)):
        return x

    if not isinstance(x, str):
        return None

    text = x.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text).strip()

    obj = _try_parse(text)
    if obj is not None:
        return obj

    m_obj = re.search(r"\{.*\}", text, flags=re.S)
    if m_obj:
        obj = _try_parse(m_obj.group(0))
        if obj is not None:
            return obj

    m_list = re.search(r"\[.*\]", text, flags=re.S)
    if m_list:
        obj = _try_parse(m_list.group(0))
        if obj is not None:
            return obj

    return None


def normalize_problem_item(x: Dict[str, Any]) -> Dict[str, str]:
    return {
        "level1": str(x.get("level1", "")).strip(),
        "level1_5": str(x.get("level1_5", "")).strip(),
        "level2": str(x.get("level2", "")).strip(),
    }


def normalize_output(x: Any) -> Dict[str, List[Dict[str, str]]]:
    """
    Accept either:
    - {"problems": [...]}
    - [...]
    - json string of either of above
    Return unified dict form.
    """
    obj = extract_json_like(x)
    if isinstance(obj, list):
        return {"problems": [normalize_problem_item(p) for p in obj if isinstance(p, dict)]}
    if isinstance(obj, dict):
        probs = obj.get("problems", [])
        if isinstance(probs, list):
            return {"problems": [normalize_problem_item(p) for p in probs if isinstance(p, dict)]}
    return {"problems": []}


def load_schema(schema_path: str):
    data = load_json(schema_path)
    items = data.get("items", [])
    l15_to_l1 = {}
    l2_to_l15 = {}
    triplets = set()
    for x in items:
        l1 = str(x["level1"]).strip()
        l15 = str(x["level1_5"]).strip()
        l2 = str(x["level2"]).strip()
        l15_to_l1[l15] = l1
        l2_to_l15[l2] = l15
        triplets.add((l1, l15, l2))
    return {"l15_to_l1": l15_to_l1, "l2_to_l15": l2_to_l15, "triplets": triplets}


def match_score(pred_item: Dict[str, str], gold_item: Dict[str, str]) -> int:
    if pred_item["level2"] == gold_item["level2"] and pred_item["level2"]:
        return 3
    if pred_item["level1_5"] == gold_item["level1_5"] and pred_item["level1"] == gold_item["level1"]:
        return 2
    if pred_item["level1"] == gold_item["level1"] and pred_item["level1"]:
        return 1
    return 0


def greedy_match(pred_items, gold_items):
    pairs = []
    used_pred = set()
    used_gold = set()

    cands = []
    for i, p in enumerate(pred_items):
        for j, g in enumerate(gold_items):
            s = match_score(p, g)
            if s > 0:
                cands.append((s, i, j))
    cands.sort(key=lambda x: (-x[0], x[1], x[2]))

    for s, i, j in cands:
        if i in used_pred or j in used_gold:
            continue
        used_pred.add(i)
        used_gold.add(j)
        pairs.append((i, j))

    unmatched_pred = [i for i in range(len(pred_items)) if i not in used_pred]
    unmatched_gold = [j for j in range(len(gold_items)) if j not in used_gold]
    return pairs, unmatched_pred, unmatched_gold


def f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def hierarchical_consistency(pred_items, schema):
    if not pred_items:
        return 0.0
    ok = 0
    for p in pred_items:
        l1, l15, l2 = p["level1"], p["level1_5"], p["level2"]
        if schema["l15_to_l1"].get(l15) == l1 and schema["l2_to_l15"].get(l2) == l15:
            ok += 1
    return ok / len(pred_items)


def exact_match_multiset(pred_items, gold_items):
    pred_keys = sorted((x["level1"], x["level1_5"], x["level2"]) for x in pred_items)
    gold_keys = sorted((x["level1"], x["level1_5"], x["level2"]) for x in gold_items)
    return int(pred_keys == gold_keys)


def problem_exact_match_count(pred_items, gold_items):
    pred_used = [False] * len(pred_items)
    correct = 0
    for g in gold_items:
        for i, p in enumerate(pred_items):
            if pred_used[i]:
                continue
            if p["level1"] == g["level1"] and p["level1_5"] == g["level1_5"] and p["level2"] == g["level2"]:
                pred_used[i] = True
                correct += 1
                break
    return correct, len(gold_items)


def evaluate(pred_file: str, schema_file: str, fixed_out: str = ""):
    schema = load_schema(schema_file)

    n_samples = 0
    bad_pred_count = 0
    bad_gold_count = 0

    tp_l1 = fp_l1 = fn_l1 = 0
    tp_l15 = fp_l15 = fn_l15 = 0
    tp_l2 = fp_l2 = fn_l2 = 0

    count_correct = 0
    hier_scores = []
    complaint_exact_total = 0
    problem_exact_correct = 0
    problem_exact_total = 0

    fixed_rows = []

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_samples += 1

            pred_norm = normalize_output(row.get("prediction_text"))
            gold_norm = normalize_output(row.get("gold"))

            if not pred_norm["problems"] and str(row.get("prediction_text", "")).strip():
                bad_pred_count += 1
            if not gold_norm["problems"] and str(row.get("gold", "")).strip():
                bad_gold_count += 1

            pred_items = pred_norm["problems"]
            gold_items = gold_norm["problems"]

            if len(pred_items) == len(gold_items):
                count_correct += 1

            pairs, unmatched_pred, unmatched_gold = greedy_match(pred_items, gold_items)

            for i, j in pairs:
                p, g = pred_items[i], gold_items[j]
                if p["level1"] == g["level1"]:
                    tp_l1 += 1
                else:
                    fp_l1 += 1
                    fn_l1 += 1

                if p["level1_5"] == g["level1_5"]:
                    tp_l15 += 1
                else:
                    fp_l15 += 1
                    fn_l15 += 1

                if p["level2"] == g["level2"]:
                    tp_l2 += 1
                else:
                    fp_l2 += 1
                    fn_l2 += 1

            fp_l1 += len(unmatched_pred); fn_l1 += len(unmatched_gold)
            fp_l15 += len(unmatched_pred); fn_l15 += len(unmatched_gold)
            fp_l2 += len(unmatched_pred); fn_l2 += len(unmatched_gold)

            hier_scores.append(hierarchical_consistency(pred_items, schema))

            c, t = problem_exact_match_count(pred_items, gold_items)
            problem_exact_correct += c
            problem_exact_total += t
            complaint_exact_total += exact_match_multiset(pred_items, gold_items)

            fixed_rows.append({
                "id": row.get("id", ""),
                "prediction_text": row.get("prediction_text", ""),
                "prediction_norm": pred_norm,
                "gold_norm": gold_norm
            })

    f1_l1 = f1_from_counts(tp_l1, fp_l1, fn_l1)
    f1_l15 = f1_from_counts(tp_l15, fp_l15, fn_l15)
    f1_l2 = f1_from_counts(tp_l2, fp_l2, fn_l2)
    weighted_tag_score = 0.2 * f1_l1 + 0.3 * f1_l15 + 0.5 * f1_l2

    metrics = {
        "n_samples": n_samples,
        "bad_pred_count": bad_pred_count,
        "bad_gold_count": bad_gold_count,
        "Count Accuracy": round(count_correct / n_samples if n_samples else 0.0, 6),
        "F1_level1": round(f1_l1, 6),
        "F1_level1_5": round(f1_l15, 6),
        "F1_level2": round(f1_l2, 6),
        "Weighted Tag Score": round(weighted_tag_score, 6),
        "Hierarchical Consistency": round(sum(hier_scores) / len(hier_scores) if hier_scores else 0.0, 6),
        "Problem Exact Match": round(problem_exact_correct / problem_exact_total if problem_exact_total else 0.0, 6),
        "Complaint Exact Match": round(complaint_exact_total / n_samples if n_samples else 0.0, 6),
    }

    if fixed_out:
        Path(fixed_out).parent.mkdir(parents=True, exist_ok=True)
        with open(fixed_out, "w", encoding="utf-8") as w:
            for row in fixed_rows:
                w.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True, type=str)
    parser.add_argument("--schema_file", required=True, type=str)
    parser.add_argument("--out_file", required=True, type=str)
    parser.add_argument("--fixed_out", default="", type=str)
    args = parser.parse_args()

    metrics = evaluate(args.pred_file, args.schema_file, args.fixed_out)
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] metrics saved to: {args.out_file}")


if __name__ == "__main__":
    main()
