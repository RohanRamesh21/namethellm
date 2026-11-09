"""Frozen mean-pooled embeddings from chosen layers of an encoder/decoder.

Usage: python3 embed.py MODEL_PATH_OR_ID OUT.npy LAYERS DTYPE
  LAYERS: comma list of hidden_state indices (0 = embeddings, -1 = last)
  DTYPE:  fp32 | bf16
Saves array of shape (n_rows, n_layers, hidden) in train.csv row order.
"""
import sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import load

path, out, layers, dtype = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
d, _, _, _ = load()
texts = d.text.tolist()

tok = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path, dtype=torch.bfloat16 if dtype == "bf16" else torch.float32).cuda().eval()

order = np.argsort([len(t) for t in texts])
res = np.zeros((len(texts), len(layers), model.config.hidden_size), dtype=np.float32)
bs = 16
with torch.no_grad():
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        for li, L in enumerate(layers):
            h = hs[L].float()
            res[idx, li] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        if (i // bs) % 40 == 0:
            print(f"{i}/{len(order)}", flush=True)
np.save(out, res)
print("saved", out, res.shape)
