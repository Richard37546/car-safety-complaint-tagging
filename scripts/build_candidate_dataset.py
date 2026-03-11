#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


CANDIDATE_INSTRUCTION = """你是汽车安全投诉结构化分析助手。
请从下面的投诉文本中抽取所有安全相关问题项，并输出标准 JSON。

要求：
1. 只保留安全相关问题，不要输出赔偿诉求、售后态度、保险、服务纠纷等非安全问题。
2. 你只能从下面给出的【候选标签】中选择，不允许改写标签名称，不允许输出候选标签之外的标签。
3. 如果一条投诉包含多个独立的安全问题，请全部输出。
4. 如果文本证据不足以支持某个候选标签，不要输出该问题项。
5. 同一个安全问题如果只是重复描述或换一种说法，不要重复输出。
6. 输出必须是严格 JSON，格式如下：
{"problems":[{"level1":"...","level1_5":"...","level2":"..."}]}
7. 不要输出任何解释、分析过程或额外文字。"""


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {e}")
    return rows


def write_jsonl(rows, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_retrieval_map(detail_file: str):
    rows = read_jsonl(detail_file)
    mp = {}
    for row in rows:
        sample_id = row.get("id")
        if sample_id:
            mp[sample_id] = row
    return mp


def candidate_item_to_text(x: dict) -> str:
    label_text = str(x.get("label_text", "")).strip()
    if label_text:
        return label_text

    l1 = str(x.get("level1", "")).strip()
    l15 = str(x.get("level1_5", "")).strip()
    l2 = str(x.get("level2", "")).strip()
    return f"{l1} / {l15} / {l2}"


def build_candidate_block(detail_row: dict, candidate_key: str, candidate_topn: int) -> str:
    cands = detail_row.get(candidate_key, [])
    if not isinstance(cands, list) or not cands:
        raise ValueError(f"id={detail_row.get('id')} 找不到候选字段 {candidate_key}，或其为空")

    lines = []
    for idx, cand in enumerate(cands[:candidate_topn], start=1):
        lines.append(f"{idx}. {candidate_item_to_text(cand)}")
    return "\n".join(lines)


def build_new_input(old_input: str, candidate_block: str) -> str:
    return f"""【候选标签】
{candidate_block}

【投诉文本】
{old_input}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="原始 train/dev/test_gold.jsonl")
    parser.add_argument("--retrieval_detail", required=True, help="对应 split 的 retrieval_eval_detail.jsonl")
    parser.add_argument("--candidate_key", default="hybrid_top30", help="例如 hybrid_top30")
    parser.add_argument("--candidate_topn", type=int, default=15, help="实际放入样本的候选数")
    parser.add_argument("--out_file", required=True, help="输出的新 jsonl")
    args = parser.parse_args()

    src_rows = read_jsonl(args.src_file)
    retrieval_map = build_retrieval_map(args.retrieval_detail)

    out_rows = []
    miss = 0

    for row in src_rows:
        sample_id = row.get("id")
        if sample_id not in retrieval_map:
            miss += 1
            continue

        candidate_block = build_candidate_block(
            retrieval_map[sample_id],
            candidate_key=args.candidate_key,
            candidate_topn=args.candidate_topn
        )

        new_row = {
            "id": row["id"],
            "instruction": CANDIDATE_INSTRUCTION,
            "input": build_new_input(str(row["input"]), candidate_block),
            "output": row["output"]
        }
        out_rows.append(new_row)

    write_jsonl(out_rows, args.out_file)

    print(json.dumps({
        "src_file": args.src_file,
        "retrieval_detail": args.retrieval_detail,
        "candidate_key": args.candidate_key,
        "candidate_topn": args.candidate_topn,
        "out_file": args.out_file,
        "num_input_rows": len(src_rows),
        "num_output_rows": len(out_rows),
        "missing_ids": miss
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()