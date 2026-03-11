import ast
import json
import re
from pathlib import Path
import argparse


def extract_json_obj(text):
    if not isinstance(text, str):
        return None

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text).strip()

    for candidate in [text]:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass

    match_obj = re.search(r"\{.*\}", text, flags=re.S)
    if match_obj:
        candidate = match_obj.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass

    match_list = re.search(r"\[.*\]", text, flags=re.S)
    if match_list:
        candidate = match_list.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass

    return None


def normalize_problem_item(x):
    return {
        "level1": str(x.get("level1", "")).strip(),
        "level1_5": str(x.get("level1_5", "")).strip(),
        "level2": str(x.get("level2", "")).strip(),
    }


def normalize_output(obj):
    if isinstance(obj, list):
        return {"problems": [normalize_problem_item(p) for p in obj if isinstance(p, dict)]}

    if isinstance(obj, dict):
        problems = obj.get("problems", [])
        if isinstance(problems, list):
            return {"problems": [normalize_problem_item(p) for p in problems if isinstance(p, dict)]}

    return {"problems": []}


def key_of_problem(x):
    return (x["level1"], x["level1_5"], x["level2"])


def complaint_exact_match(pred_items, gold_items):
    pred_keys = sorted([key_of_problem(x) for x in pred_items])
    gold_keys = sorted([key_of_problem(x) for x in gold_items])
    return pred_keys == gold_keys


def has_duplicate_problems(pred_items):
    keys = [key_of_problem(x) for x in pred_items]
    return len(keys) != len(set(keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bad_pred = []
    count_wrong = []
    duplicate_pred = []
    complaint_wrong = []

    total = 0

    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            row = json.loads(line)

            pred_raw = extract_json_obj(row["prediction_text"])
            gold_raw = row["gold"]

            pred_bad = pred_raw is None
            pred_obj = {"problems": []} if pred_bad else normalize_output(pred_raw)
            gold_obj = normalize_output(gold_raw if not isinstance(gold_raw, str) else extract_json_obj(gold_raw))

            pred_items = pred_obj["problems"]
            gold_items = gold_obj["problems"]

            record = {
                "id": row.get("id"),
                "prediction_text": row.get("prediction_text"),
                "pred_parsed": pred_obj,
                "gold": gold_obj,
                "pred_count": len(pred_items),
                "gold_count": len(gold_items),
            }

            if pred_bad:
                bad_pred.append(record)

            if len(pred_items) != len(gold_items):
                count_wrong.append(record)

            if has_duplicate_problems(pred_items):
                duplicate_pred.append(record)

            if not complaint_exact_match(pred_items, gold_items):
                complaint_wrong.append(record)

    def dump(name, data):
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        return path

    p1 = dump("bad_pred.jsonl", bad_pred)
    p2 = dump("count_wrong.jsonl", count_wrong)
    p3 = dump("duplicate_pred.jsonl", duplicate_pred)
    p4 = dump("complaint_wrong.jsonl", complaint_wrong)

    summary = {
        "total": total,
        "bad_pred": len(bad_pred),
        "count_wrong": len(count_wrong),
        "duplicate_pred": len(duplicate_pred),
        "complaint_wrong": len(complaint_wrong),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] bad_pred -> {p1}")
    print(f"[OK] count_wrong -> {p2}")
    print(f"[OK] duplicate_pred -> {p3}")
    print(f"[OK] complaint_wrong -> {p4}")


if __name__ == "__main__":
    main()