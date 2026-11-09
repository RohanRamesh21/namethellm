"""Do the frozen prompt-centred models add to the LoRA blend? Usage: python3 blend_lora.py RESULTS_RUN_DIR SCRATCH_DIR
Groups for centring and assignment are estimated from text (Ward on Qwen-emb last layer, K = rows/18)."""
import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from common import load, bacc, center_by_group, assign_by_group

run, S = Path(sys.argv[1]), sys.argv[2]
d, y, true_g, classes = load()
q = np.load(f"{S}/emb_qwen.npy")[:, 1]
est = AgglomerativeClustering(n_clusters=round(len(d) / 18), linkage="ward").fit(normalize(q - q.mean(0))).labels_

lora = np.mean([np.load(p)["logp"].astype(np.float64) for p in sorted(run.glob("runs/*/pred.npz"))], axis=0)
ng = np.load(run / "ngram.npz")["logp"].astype(np.float64)
fq, fb = np.load(f"{S}/oof_qwen_L0_C0.01_cent.npy"), np.load(f"{S}/oof_bert_L0_C0.01_cent.npy")
z = lambda a: a / a.std()
parts = {"lora": z(center_by_group(lora, est)), "ngram": z(center_by_group(ng, est)), "frozen_qwen": z(fq), "frozen_bert": z(fb)}

def show(label, S_):
    print(f"  {label:<44} argmax {bacc(y, S_.argmax(1)):.3f} | assign {bacc(y, assign_by_group(S_, est)):.3f}")

print("scores centred per estimated prompt; assign = one model per prompt")
show("LoRA mean of 4 runs", parts["lora"])
show("LoRA + n-gram", parts["lora"] + parts["ngram"])
show("LoRA + frozen qwen + frozen bert", parts["lora"] + parts["frozen_qwen"] + parts["frozen_bert"])
show("LoRA + n-gram + frozen qwen + frozen bert", sum(parts.values()))
show("LoRA x2 + n-gram + frozen qwen + frozen bert", sum(parts.values()) + parts["lora"])
