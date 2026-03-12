# Car Safety Complaint Tagging

面向真实汽车安全投诉文本的多问题层级标签抽取项目。  
目标是从一条自由描述的投诉文本中，抽取所有安全相关问题，并映射到标准三级标签体系。

最终最佳方案采用：

- **Hybrid Retrieval** 先从标签体系中检索最相关的候选标签
- **Top-k Candidate-aware Prompting** 推理时把 top-k 候选标签加入 prompt(instruction)，约束模型只能在候选中选择
- **Candidate-aware LoRA SFT**  训练时也使用带候选标签的数据，让模型学会更稳定地在候选集合中做结构化选择

相比纯自由生成方式，最终方案在层级一致性、精确匹配和整体稳定性上都有明显提升。

---

## 1. Project Overview

本项目聚焦于真实汽车安全投诉场景下的**闭集标准标签抽取**问题。

给定一条投诉文本，模型需要抽取其中所有安全相关问题项，并输出标准化的三级标签结果：

- `level1`
- `level1_5`
- `level2`

这不是普通的单标签分类任务，而是一个同时具备以下特点的 NLP 工程任务：

- **多问题抽取**：一条投诉可能包含多个安全问题
- **层级标签预测**：标签存在 `level1 -> level1_5 -> level2` 的层级关系
- **闭集标准标签约束**：输出必须来自固定 schema(已冻结的标签体系)，而不是自由生成
- **真实行业数据**：数据来自真实汽车投诉数据，经人工筛选和清洗

---

## 2. Task Definition

### Input
每条样本的输入字段包括：

- 品牌
- 问题简述
- 详细描述

示例：

```text
品牌：福田时代
问题简述：制动系
详细描述：福田祥菱V3产品质量控告
```

### Output
输出多个安全问题项，每个问题项为一个标准三元组：

```
{
  "problems": [
    {
      "level1": "制动系统",
      "level1_5": "制动系统机械部件",
      "level2": "制动失灵"
    }
  ]
}
```

---

## 3. Dataset

原始数据来自 Excel 表格中的真实汽车投诉数据。
经过人工清洗、字段映射、标签对齐与样本筛选后，最终得到 597 条可用样本。

数据切分如下：
- train: 418
- dev: 60
- test_gold: 119

说明：当前版本不做 severity 分析 (严重性)

---

## 4. 这个项目的难点

这个任务的难点主要体现在：

### 一条投诉可能包含多个问题
- 不是单标签分类，而是多问题结构化抽取

### 投诉文本是自由表达，标签是标准表达

- 用户描述通常是口语化、现象化的
- 标准标签是规范化、闭集、层级化的

### 纯生成容易“语义接近但标签不标准”

- 模型可能输出近义标签、改写标签、泛化标签
- 这会直接影响指标 exact match 和层级一致性

### 标签体系具有严格层级约束

- 除了叶子标签要对，父层级也必须匹配
- 因此仅靠“差不多的语义”是不够的

---

## 5. Benchmark

项目统一使用同一套评测逻辑，对 baseline、自由生成 LoRA、candidate-aware inference、candidate-aware SFT 进行公平比较。

###核心指标包括：

Count Accuracy、F1_level1、F1_level1、F1_level1_5、F1_level2、
Weighted Tag Score、Hierarchical Consistency、
Problem Exact Match、Complaint Exact Match

其中： Weighted Tag Score = 0.2 * F1_level1 + 0.3 * F1_level1_5 + 0.5 * F1_level2

这些指标共同衡量：问题数量是否正确、层级标签是否精确、结构是否自洽、整条投诉是否完全匹配验证集

---

## 6. 方法的探索过程

### Stage 1: Free-generation baseline(先直接让模型自由生成标签，看看能做到什么程度)

首先采用 Qwen2.5-3B-Instruct 进行自由生成式结构化抽取，并进一步使用 LoRA 微调。

这一阶段暴露出几个核心问题：

- 标签名称容易被改写
- 近义替换较多
- 多问题样本数量控制不稳
- 层级一致性较差
- Exact Match 较低

### Stage 2: Candidate Retrieval(发现自由生成不稳后，先从标签体系里找出一批可能正确的候选标签)

通过对错误数据分析发现，这个任务本质上更接近于闭集标签选择问题，而不是开放生成问题。

因此项目将任务重构为：

- 先检索候选标签，再进行受限选择输出

对标签检索方法进行比较后，得到如下结论：

- BM25 < Dense < Hybrid

说明该任务不仅依赖词面匹配，也强依赖语义理解，最终采用 Hybrid Retrieval。

### Stage 3: Top-k Candidate-aware Prompting（把候选标签放进 prompt，让模型在候选中做受限选择）

在推理阶段，将候选标签注入 prompt（instruction)，并约束模型只能从候选集合中进行选择。

进一步对不同 top_k 进行实验比较后，发现：

- top10 有明显提升
- top15 效果最佳
- top20 / top30 继续增大候选集后，没有带来更好的最终表现

因此最终选定：

- Hybrid Retrieval + top15 candidates

### Stage 4: Candidate-aware LoRA SFT（不仅推理时给候选标签，训练时也让模型学会如何在候选中做选择）

在 inference-only 的 candidate-aware prompt 已经显著提升后，进一步将 train/dev 数据也构造为 candidate-aware 形式，并进行 LoRA 微调。

最终得到当前最佳版本。

---


## 7. Final Pipeline

```
Complaint Text
    ↓
Hybrid Retrieval
    ↓
Top-15 Candidate Labels
    ↓
Candidate-aware Prompt / Candidate-aware SFT
    ↓
Structured JSON Output
```

---

## 8. Results

最终最佳方案为：

Hybrid Retrieval + top15 candidates + Candidate-aware LoRA SFT


### Main Results

| Method | Count Acc. | F1_level2 | Weighted Tag Score | Hierarchical Consistency | Problem EM | Complaint EM | bad_pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (free generation) | 0.6581 | 0.0322 | 0.0418 | 0.0000 | 0.0000 | 0.0000 | 0 |
| LoRA (free generation) | 0.6325 | 0.0519 | 0.1357 | 0.0299 | 0.0063 | 0.0000 | 4 |
| LoRA + candidate-aware inference (Hybrid + top15) | 0.7265 | 0.3942 | 0.4788 | 0.5983 | 0.2642 | 0.2564 | 6 |
| **LoRA + candidate-aware SFT (Hybrid + top15)** | **0.7436** | **0.5017** | **0.5863** | **0.8903** | **0.4403** | **0.4274** | **0** |

### Key Finding

相比纯自由生成方式，这一方案明显提升了：

- 标签输出的标准化程度

- 层级结构一致性

- 多问题抽取的准确性

- 整条投诉的完全匹配率

说明对于该类闭集层级标签抽取任务，
“候选标签检索 + 受限选择 + candidate-aware SFT” 明显优于纯自由生成。

---

## 9. Repository Structure

```
.
├── README.md
├── configs/
├── data/
│   ├── schema/
│   └── samples/
├── docs/
├── results/
└── scripts/

```
### 说明：

- data/schema/：标签体系文件
- scripts/：检索、推理、评测、错误分析脚本
- configs/：LoRA / LLaMA-Factory 配置
- results/：关键实验结果
- docs/：任务定义、字段映射、方法说明等补充文档

---


## 10. Key Scripts

- retrieval_eval.py
候选标签检索实验（BM25 / Dense / Hybrid）

- build_candidate_dataset.py
构造 candidate-aware train/dev/test 数据

- run_predict_lora_candidate.py
candidate-aware 推理脚本（LoRA）

- run_baseline_candidate.py
baseline 推理脚本

- eval_structured_consistent.py
统一评测脚本

- extract_error_cases.py
错误样本提取与分析

---


## 11. Notes

- 当前版本不做 severity

- 项目重点是闭集标准标签抽取，而不是开放生成

---

## 12. Future Work

- **更好的候选标签排序**  
  当前系统已经能够检索出一批候选标签，后续可以继续优化这些候选标签的排序，让真正正确的标签排得更靠前。

- **使用标签编号替代直接输出标签文本**  
  当前模型直接生成中文标签名称，后续可以尝试让模型输出标签 ID，再由程序映射回标准标签，以减少标签改写和格式错误。

- **更细粒度的错误分析与可视化**  
  后续可以进一步分析不同标签类型、不同问题数量场景下的错误模式，并用图表形式展示结果。

- **更强的检索增强结构化抽取流程**  
  当前方案已经验证了“检索候选标签 + 受限选择”的有效性，后续可以继续增强检索、排序和结构化解码能力。


## 13. Model

Fine-tuned LoRA weights are available on Hugging Face:

https://huggingface.co/Richard37546/car-safety-complaint-lora









