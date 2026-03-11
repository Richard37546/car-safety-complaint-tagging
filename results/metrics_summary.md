# Metrics Summary

## Final Best Result

Final best setting:

- Hybrid Retrieval
- top15 candidate labels
- Candidate-aware LoRA SFT

Key metrics:

- Count Accuracy: 0.74359
- F1_level2: 0.501718
- Weighted Tag Score: 0.586254
- Hierarchical Consistency: 0.890313
- Problem Exact Match: 0.440252
- Complaint Exact Match: 0.42735
- bad_pred_count: 0

## Compared Settings

| Method | Count Acc. | F1_level2 | Weighted Tag Score | Hierarchical Consistency | Problem EM | Complaint EM | bad_pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (free generation) | 0.6581 | 0.0322 | 0.0418 | 0.0000 | 0.0000 | 0.0000 | 0 |
| LoRA (free generation) | 0.6325 | 0.0519 | 0.1357 | 0.0299 | 0.0063 | 0.0000 | 4 |
| LoRA + candidate-aware inference (Hybrid + top15) | 0.7265 | 0.3942 | 0.4788 | 0.5983 | 0.2642 | 0.2564 | 6 |
| **LoRA + candidate-aware SFT (Hybrid + top15)** | **0.7436** | **0.5017** | **0.5863** | **0.8903** | **0.4403** | **0.4274** | **0** |
