"""Blend out-of-fold scores from several models, with and without prompt removal.

Usage: python3 blend.py SPEC [SPEC ...]
  SPEC = name=RAW.npy|CENT.npy   CENT may be the word CENTER to centre RAW's scores per prompt.
Each model's scores are divided by their overall std so no model dominates by scale.
Prints balanced accuracy for single models and for the blend of all, raw vs prompt-removed
(and with one-model-per-prompt assignment, valid only if the hidden test has that structure).
"""
import sys
import numpy as np

from common import load, bacc, center_by_group, assign_by_group

d, y, groups, classes = load()
specs = {}
for s in sys.argv[1:]:
    name, files = s.split("=")
    raw_f, cent_f = files.split("|")
    raw = np.load(raw_f)
    cent = center_by_group(raw, groups) if cent_f == "CENTER" else np.load(cent_f)
    specs[name] = (raw / raw.std(), cent / cent.std())


def show(label, S):
    print(f"  {label:<34} argmax {bacc(y, S.argmax(1)):.3f} | assign {bacc(y, assign_by_group(S, groups)):.3f}")


print("single models")
for n, (r, c) in specs.items():
    show(f"{n} raw", r)
    show(f"{n} prompt-removed", c)
print("blend of all (equal weights)")
show("raw", sum(r for r, _ in specs.values()))
show("prompt-removed", sum(c for _, c in specs.values()))
