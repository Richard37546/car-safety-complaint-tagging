#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="从 label_schema_v1.json 构建可检索标签库")
    parser.add_argument("--schema_file", required=True, help="schema json 文件路径")
    parser.add_argument("--out_file", required=True, help="输出 jsonl 文件路径")
    args = parser.parse_args()

    data = load_json(args.schema_file)
    items = data.get("items", [])

    if not isinstance(items, list):
        raise ValueError("schema 文件格式不正确：缺少 items 列表")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    rows = []

    for idx, item in enumerate(items, start=1):
        level1 = str(item.get("level1", "")).strip()
        level1_5 = str(item.get("level1_5", "")).strip()
        level2 = str(item.get("level2", "")).strip()

        if not level1 or not level1_5 or not level2:
            continue

        key = (level1, level1_5, level2)
        if key in seen:
            continue
        seen.add(key)

        label_id = f"L{len(rows)+1:04d}"
        label_text = f"{level1} / {level1_5} / {level2}"

        rows.append({
            "label_id": label_id,
            "level1": level1,
            "level1_5": level1_5,
            "level2": level2,
            "label_text": label_text
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "schema_file": args.schema_file,
        "out_file": str(out_path),
        "num_labels": len(rows)
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()