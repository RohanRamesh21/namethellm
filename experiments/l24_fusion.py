"""Turn l24_worker.py output into log-probabilities. For rows inside an estimated prompt group of 2-18 passages use the CENTRED head
(raw logits - group mean + the head's bias); other rows use the plain head. Average the log-probs over the fold adapters.
Out-of-fold (experiments/l24_blend_local.py): centred head alone beat plain (0.665 vs 0.615) and beat any plain/centred mix when blended with the native expert."""
import numpy as np
from scipy.special import log_softmax


def l24_logp(npz_path, groups, min_group=2, max_group=18):
    z = np.load(npz_path, allow_pickle=False)
    plain, craw, cbias = z['plain'].astype(np.float64), z['centred_raw'].astype(np.float64), z['centred_bias'].astype(np.float64)
    groups = np.asarray(groups)
    if plain.shape[1] != len(groups): raise ValueError('group labels do not match the scored passages')
    inv = np.unique(groups, return_inverse=True)[1]; count = np.bincount(inv); eligible = ((count[inv] >= min_group) & (count[inv] <= max_group))[:, None]
    out = []
    for a in range(plain.shape[0]):
        mean = np.zeros((len(count), plain.shape[2])); np.add.at(mean, inv, craw[a]); mean /= count[:, None]
        out.append(log_softmax(np.where(eligible, craw[a] - mean[inv] + cbias[a], plain[a]), axis=1))
    return log_softmax(np.mean(out, axis=0), axis=1)
