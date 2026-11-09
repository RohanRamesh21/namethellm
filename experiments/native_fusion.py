"""Combine the native expert's raw head logits (native_worker.py output) with our parent scores.

native_logp(): per fold, calibrate the plain head and the prompt-centred head with the author's stored scales, centre the second inside the
estimated prompt group (groups of 2-18 passages only; other rows keep the plain head), mix them at the fold's stored weight, then average the
five folds' log-probabilities.
blend_parent_native(): z-scored weighted sum, the same way the rest of the pipeline blends.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import log_softmax


def native_logp(npz_path, assets, groups, labels):
    z = np.load(npz_path, allow_pickle=False)
    saved = z['labels'].tolist()
    if sorted(saved) != sorted(labels): raise ValueError('class names differ between the expert and the pipeline')
    order = [saved.index(l) for l in labels]
    plain, cent, bias, folds = z['plain'][:, :, order], z['centred'][:, :, order], z['centred_bias'][:, order], z['folds'].tolist()
    groups = np.asarray(groups)
    if plain.shape[1] != len(groups): raise ValueError('group labels do not match the scored passages')
    inv = np.unique(groups, return_inverse=True)[1]; count = np.bincount(inv); eligible = (count[inv] >= 2) & (count[inv] <= 18)
    out = []
    for i, fold in enumerate(folds):
        cal = json.loads((Path(assets) / 'calibration' / f'fold{fold}.json').read_text())['models']['qwen8']
        if not cal['centered_head_trained'] or cal['outer_labels_used']: raise ValueError(f'fold {fold}: unexpected calibration record')
        mean = np.zeros((len(count), cent.shape[2])); np.add.at(mean, inv, cent[i]); mean /= count[:, None]
        a = log_softmax(plain[i] * cal['plain_scale'], axis=1)
        b = log_softmax((cent[i] - mean[inv] + bias[i]) * cal['centered_scale'], axis=1)
        w = np.where(eligible, cal['selected_centered_weight'], 0.0)[:, None]
        out.append(log_softmax((1 - w) * a + w * b, axis=1))
    return np.mean(out, axis=0)


def blend_parent_native(parent, native, native_weight):
    z = lambda a: a / a.std()
    return (1 - native_weight) * z(parent) + native_weight * z(native)
