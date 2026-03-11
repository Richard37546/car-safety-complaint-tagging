import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


DEFAULT_BASE_MODEL = "/data/syl/QwenQwen2.5_3B_Instruct"
DEFAULT_ADAPTER_PATH = "/data/syl/lora_QwenQwen2.5_3B_Instruct/lora_model"
DEFAULT_TEST_FILE = "/data/syl/tiny_practice/data/v1_best/test_gold.jsonl"
DEFAULT_RETRIEVAL_DETAIL = "/data/syl/tiny_practice/results/retrieval_eval_hybrid/retrieval_eval_detail.jsonl"
DEFAULT_OUT_FILE = "/data/syl/tiny_practice/results/lora_predictions_candidate.jsonl"
DEFAULT_CANDIDATE_KEY = "hybrid_top30"
DEFAULT_CANDIDATE_TOPN = 10
DEFAULT_MAX_NEW_TOKENS = 512


def ensure_parent(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: str):
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


def build_retrieval_map(detail_file: str):
    rows = load_jsonl(detail_file)
    mp = {}
    for row in rows:
        sample_id = row.get("id")
        if not sample_id:
            continue
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


def build_prompt(input_text: str, candidate_block: str) -> str:
    return f"""你是汽车安全投诉结构化分析助手。
请从下面的投诉文本中抽取所有安全相关问题项，并输出标准 JSON。

要求：
1. 只保留安全相关问题，不要输出赔偿诉求、售后态度、保险、服务纠纷等非安全问题。
2. 你只能从下面给出的【候选标签】中选择，不允许改写标签名称，不允许输出候选标签之外的标签。
3. 如果一条投诉包含多个独立的安全问题，请全部输出。
4. 如果文本证据不足以支持某个候选标签，不要输出该问题项。
5. 同一个安全问题如果只是重复描述或换一种说法，不要重复输出。
6. 输出必须是严格 JSON，格式如下：
{{"problems":[{{"level1":"...","level1_5":"...","level2":"..."}}]}}
7. 不要输出任何解释、分析过程或额外文字。

【候选标签】
{candidate_block}

【投诉文本】
{input_text}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, type=str)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER_PATH, type=str)
    parser.add_argument("--test_file", default=DEFAULT_TEST_FILE, type=str)
    parser.add_argument("--retrieval_detail", default=DEFAULT_RETRIEVAL_DETAIL, type=str)
    parser.add_argument("--out_file", default=DEFAULT_OUT_FILE, type=str)
    parser.add_argument("--candidate_key", default=DEFAULT_CANDIDATE_KEY, type=str,
                        help="例如 hybrid_top30 / dense_top30 / bm25_top30")
    parser.add_argument("--candidate_topn", default=DEFAULT_CANDIDATE_TOPN, type=int,
                        help="实际放进 prompt 的候选数量，例如 10 或 15")
    parser.add_argument("--max_new_tokens", default=DEFAULT_MAX_NEW_TOKENS, type=int)
    args = parser.parse_args()

    ensure_parent(args.out_file)
    retrieval_map = build_retrieval_map(args.retrieval_detail)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    n = 0
    with open(args.test_file, "r", encoding="utf-8") as f, open(args.out_file, "w", encoding="utf-8") as w:
        for line in f:
            item = json.loads(line)
            sample_id = item["id"]

            if sample_id not in retrieval_map:
                raise KeyError(f"id={sample_id} 在 retrieval_detail 中不存在")

            candidate_block = build_candidate_block(
                retrieval_map[sample_id],
                candidate_key=args.candidate_key,
                candidate_topn=args.candidate_topn
            )
            prompt = build_prompt(item["input"], candidate_block)

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )

            pred_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            record = {
                "id": sample_id,
                "prediction_text": pred_text,
                "gold": item["output"]
            }
            w.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] saved {n} predictions to: {args.out_file}")


if __name__ == "__main__":
    main()