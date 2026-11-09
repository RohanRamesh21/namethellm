## Baseline (prompt-grouped 5-fold balanced accuracy)

| Model | Raw | Centred per prompt |
|---|---|---|
| ngram | 0.311 | 0.358 |
| qwen_emb | 0.350 | 0.490 |
| bert | 0.365 | 0.479 |

Blend (n-gram + Qwen-emb + BERT, all centred): **0.524**. Same blend, nothing centred: 0.412.

## Style embedders

| Model | Layers pooled | Best layer | Raw | Centred | Blend with it | Change vs baseline blend (95% interval over prompts) |
|---|---|---|---|---|---|---|
| wegmann | [6, 9, 12] | 6 | 0.344 | 0.458 | 0.552 | +0.028 (+0.017 to +0.039) |
| luar_mud | [3, 4, 6] | 3 | 0.365 | 0.492 | 0.554 | +0.030 (+0.021 to +0.040) |
| luar_crud | [3, 4, 6] | 3 | 0.357 | 0.481 | 0.556 | +0.033 (+0.023 to +0.044) |

All working style embedders added together: 0.564, change +0.041 (+0.026 to +0.056).

Best layer is picked on the same rows it is scored on, so each candidate row is slightly optimistic. An interval that includes zero means no detectable gain.
