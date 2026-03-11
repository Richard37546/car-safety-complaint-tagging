#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# =========================
# 基础 IO
# =========================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"{path} 第 {line_no} 行不是 JSON object")
            rows.append(obj)
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], path: str):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# 解析 gold output
# =========================

def _try_parse(text: str):
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(text)
        except Exception:
            pass
    return None


def extract_json_like(x: Any):
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
    obj = extract_json_like(x)
    if isinstance(obj, list):
        return {"problems": [normalize_problem_item(p) for p in obj if isinstance(p, dict)]}
    if isinstance(obj, dict):
        probs = obj.get("problems", [])
        if isinstance(probs, list):
            return {"problems": [normalize_problem_item(p) for p in probs if isinstance(p, dict)]}
    return {"problems": []}


def dedup_problems(problems: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for p in problems:
        key = (p["level1"], p["level1_5"], p["level2"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


# =========================
# query 构造
# =========================

def parse_input_text(input_text: str) -> Dict[str, str]:
    """
    尽量兼容：
    品牌：...
    问题简述：...
    详细描述：...
    """
    res = {
        "brand": "",
        "brief": "",
        "desc": "",
        "raw": input_text.strip()
    }

    if not isinstance(input_text, str):
        return res

    for line in input_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("品牌：") or line.startswith("品牌:"):
            res["brand"] = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
        elif line.startswith("问题简述：") or line.startswith("问题简述:"):
            res["brief"] = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
        elif line.startswith("详细描述：") or line.startswith("详细描述:"):
            res["desc"] = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]

    # 如果没能解析出来，就把原始文本整体塞到 desc
    if not res["brand"] and not res["brief"] and not res["desc"]:
        res["desc"] = input_text.strip()

    return res


def build_query(input_text: str) -> str:
    fields = parse_input_text(input_text)
    query = f"品牌：{fields['brand']}\n问题简述：{fields['brief']}\n详细描述：{fields['desc']}".strip()
    return query


# =========================
# 标签库读取
# =========================

def load_corpus(corpus_file: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(corpus_file)
    need_keys = {"label_id", "level1", "level1_5", "level2", "label_text"}
    for i, row in enumerate(rows, start=1):
        miss = need_keys - set(row.keys())
        if miss:
            raise ValueError(f"{corpus_file} 第 {i} 条缺少字段: {miss}")
    return rows


# =========================
# BM25（基于字符 2-gram）
# =========================

def char_ngrams(text: str, n: int = 2) -> List[str]:
    text = re.sub(r"\s+", "", text)
    if not text:
        return []
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text)-n+1)]


class BM25Retriever:
    def __init__(self, documents: List[str], ngram_n: int = 2, k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.ngram_n = ngram_n
        self.k1 = k1
        self.b = b

        self.doc_tokens = [char_ngrams(doc, ngram_n) for doc in documents]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))

        self.tf = []
        self.df = Counter()

        for toks in self.doc_tokens:
            cnt = Counter(toks)
            self.tf.append(cnt)
            for tok in cnt.keys():
                self.df[tok] += 1

        self.N = len(documents)

    def score(self, query: str) -> List[float]:
        q_tokens = char_ngrams(query, self.ngram_n)
        q_freq = Counter(q_tokens)

        scores = []
        for idx, doc_tf in enumerate(self.tf):
            dl = self.doc_lens[idx]
            s = 0.0
            for tok, qf in q_freq.items():
                if tok not in doc_tf:
                    continue
                df = self.df.get(tok, 0)
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                tf = doc_tf[tok]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                s += idf * tf * (self.k1 + 1) / max(1e-9, denom)
            scores.append(float(s))
        return scores


# =========================
# Dense Retrieval
# =========================

class DenseRetriever:
    def __init__(self, documents: List[str], model_name_or_path: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("未安装 sentence-transformers，无法使用 dense 检索")

        self.model = SentenceTransformer(model_name_or_path)
        self.documents = documents
        self.doc_emb = self.model.encode(documents, normalize_embeddings=True, show_progress_bar=False)

    def score(self, query: str) -> List[float]:
        q_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.dot(self.doc_emb, q_emb)
        return scores.tolist()


# =========================
# Hybrid (RRF)
# =========================

def reciprocal_rank_fusion(rankings: Dict[str, List[int]], c: int = 60) -> Dict[int, float]:
    """
    rankings:
      {
        "bm25": [doc_idx1, doc_idx2, ...],
        "dense": [doc_idx3, doc_idx4, ...]
      }
    """
    fused = defaultdict(float)
    for _, ranked_doc_ids in rankings.items():
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            fused[doc_id] += 1.0 / (c + rank)
    return dict(fused)


# =========================
# 评估
# =========================

def topk_from_scores(scores: List[float], k: int) -> List[int]:
    idxs = list(range(len(scores)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    return idxs[:k]


def get_gold_sets(row: Dict[str, Any]) -> Tuple[List[Dict[str, str]], set, set]:
    gold_norm = normalize_output(row.get("output"))
    gold_probs = dedup_problems(gold_norm["problems"])
    gold_level2 = set()
    gold_triplets = set()

    for p in gold_probs:
        l1 = p["level1"].strip()
        l15 = p["level1_5"].strip()
        l2 = p["level2"].strip()
        if l2:
            gold_level2.add(l2)
        if l1 and l15 and l2:
            gold_triplets.add((l1, l15, l2))

    return gold_probs, gold_level2, gold_triplets


def compute_recall(gold_set: set, pred_set: set) -> float:
    if not gold_set:
        return 0.0
    return len(gold_set & pred_set) / len(gold_set)


def compute_full_hit(gold_set: set, pred_set: set) -> bool:
    if not gold_set:
        return False
    return gold_set.issubset(pred_set)


def summarize_metrics(detail_rows: List[Dict[str, Any]], methods: List[str], topks: List[int]) -> Dict[str, Any]:
    summary = {}
    for method in methods:
        summary[method] = {}
        for k in topks:
            l2_avg_key = f"{method}_level2_recall@{k}"
            l2_full_key = f"{method}_level2_full_hit@{k}"
            tri_avg_key = f"{method}_triplet_recall@{k}"
            tri_full_key = f"{method}_triplet_full_hit@{k}"

            l2_avg = np.mean([row[l2_avg_key] for row in detail_rows]) if detail_rows else 0.0
            l2_full = np.mean([1.0 if row[l2_full_key] else 0.0 for row in detail_rows]) if detail_rows else 0.0
            tri_avg = np.mean([row[tri_avg_key] for row in detail_rows]) if detail_rows else 0.0
            tri_full = np.mean([1.0 if row[tri_full_key] else 0.0 for row in detail_rows]) if detail_rows else 0.0

            summary[method][f"level2_avg_recall@{k}"] = round(float(l2_avg), 6)
            summary[method][f"level2_full_hit@{k}"] = round(float(l2_full), 6)
            summary[method][f"triplet_avg_recall@{k}"] = round(float(tri_avg), 6)
            summary[method][f"triplet_full_hit@{k}"] = round(float(tri_full), 6)

    return summary


# =========================
# 主流程
# =========================

def main():
    parser = argparse.ArgumentParser(description="候选标签检索 Recall@K 实验")
    parser.add_argument("--corpus_file", required=True, help="label_corpus.jsonl 路径")
    parser.add_argument("--test_file", required=True, help="test_gold.jsonl 路径")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--methods", default="bm25", help="方法列表，逗号分隔：bm25,dense,hybrid")
    parser.add_argument("--dense_model", default="", help="dense 检索模型路径或名称")
    parser.add_argument("--topk", default="1,3,5,10,15", help="逗号分隔，例如 1,3,5,10,15")
    parser.add_argument("--bm25_ngram", type=int, default=2, help="BM25 使用的字符 ngram，默认 2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    topks = sorted(set(int(x.strip()) for x in args.topk.split(",") if x.strip()))
    max_k = max(topks)

    corpus = load_corpus(args.corpus_file)
    test_rows = read_jsonl(args.test_file)

    label_texts = [x["label_text"] for x in corpus]

    # 初始化 retrievers
    retrievers = {}

    if "bm25" in methods or "hybrid" in methods:
        retrievers["bm25"] = BM25Retriever(label_texts, ngram_n=args.bm25_ngram)

    if "dense" in methods or "hybrid" in methods:
        if not args.dense_model:
            raise ValueError("使用 dense 或 hybrid 时，必须提供 --dense_model")
        retrievers["dense"] = DenseRetriever(label_texts, args.dense_model)

    detail_rows = []

    for row in test_rows:
        sample_id = row.get("id", "")
        query = build_query(str(row.get("input", "")))
        gold_probs, gold_level2, gold_triplets = get_gold_sets(row)

        result_row = {
            "id": sample_id,
            "query": query,
            "gold_triplets": list(gold_triplets),
        }

        rankings = {}
        top_docs_cache = {}

        # 单方法检索
        for method in methods:
            if method == "hybrid":
                continue
            scores = retrievers[method].score(query)
            ranked_ids = topk_from_scores(scores, max_k)
            rankings[method] = ranked_ids

            top_docs = []
            for doc_id in ranked_ids:
                item = corpus[doc_id]
                top_docs.append({
                    "label_id": item["label_id"],
                    "level1": item["level1"],
                    "level1_5": item["level1_5"],
                    "level2": item["level2"],
                    "label_text": item["label_text"],
                    "score": round(float(scores[doc_id]), 6)
                })
            top_docs_cache[method] = top_docs
            result_row[f"{method}_top{max_k}"] = top_docs

            for k in topks:
                pred_level2 = {x["level2"] for x in top_docs[:k]}
                pred_triplets = {(x["level1"], x["level1_5"], x["level2"]) for x in top_docs[:k]}

                result_row[f"{method}_level2_recall@{k}"] = round(compute_recall(gold_level2, pred_level2), 6)
                result_row[f"{method}_level2_full_hit@{k}"] = compute_full_hit(gold_level2, pred_level2)
                result_row[f"{method}_triplet_recall@{k}"] = round(compute_recall(gold_triplets, pred_triplets), 6)
                result_row[f"{method}_triplet_full_hit@{k}"] = compute_full_hit(gold_triplets, pred_triplets)

        # hybrid 检索
        if "hybrid" in methods:
            required = []
            if "bm25" in rankings:
                required.append("bm25")
            if "dense" in rankings:
                required.append("dense")

            if len(required) < 2:
                raise ValueError("hybrid 需要至少两个基础方法，目前至少应包含 bm25 和 dense")

            fused_scores = reciprocal_rank_fusion({m: rankings[m] for m in required}, c=60)
            fused_ranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:max_k]

            top_docs = []
            for doc_id in fused_ranked_ids:
                item = corpus[doc_id]
                top_docs.append({
                    "label_id": item["label_id"],
                    "level1": item["level1"],
                    "level1_5": item["level1_5"],
                    "level2": item["level2"],
                    "label_text": item["label_text"],
                    "score": round(float(fused_scores[doc_id]), 6)
                })

            result_row[f"hybrid_top{max_k}"] = top_docs

            for k in topks:
                pred_level2 = {x["level2"] for x in top_docs[:k]}
                pred_triplets = {(x["level1"], x["level1_5"], x["level2"]) for x in top_docs[:k]}

                result_row[f"hybrid_level2_recall@{k}"] = round(compute_recall(gold_level2, pred_level2), 6)
                result_row[f"hybrid_level2_full_hit@{k}"] = compute_full_hit(gold_level2, pred_level2)
                result_row[f"hybrid_triplet_recall@{k}"] = round(compute_recall(gold_triplets, pred_triplets), 6)
                result_row[f"hybrid_triplet_full_hit@{k}"] = compute_full_hit(gold_triplets, pred_triplets)

        detail_rows.append(result_row)

    summary = {
        "corpus_file": args.corpus_file,
        "test_file": args.test_file,
        "num_labels": len(corpus),
        "num_samples": len(test_rows),
        "methods": methods,
        "topks": topks,
        "results": summarize_metrics(detail_rows, methods, topks)
    }

    write_jsonl(detail_rows, str(out_dir / "retrieval_eval_detail.jsonl"))
    write_json(summary, str(out_dir / "retrieval_eval_summary.json"))

    print(json.dumps({
        "detail_file": str(out_dir / "retrieval_eval_detail.jsonl"),
        "summary_file": str(out_dir / "retrieval_eval_summary.json"),
        "num_labels": len(corpus),
        "num_samples": len(test_rows),
        "methods": methods,
        "topks": topks
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()