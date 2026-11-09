"""A small version of the public 'native embedding expert': Qwen3-Embedding-0.6B + LoRA, mean-pooled hidden states at 8 depths
(a LayerNorm each), and two linear heads: a plain head and a prompt-centred head. Trained on the OFFICIAL rows by prompt fold.

Batches hold whole prompts (all of a prompt's training passages), so the centred head can be trained on features minus the
prompt's mean, the way it is applied at test time with estimated groups.

  python3 native_small.py --fold 0 --out OUT        # train on the other 4 folds, score fold 0 (never trained on)
  python3 native_small.py --fold all --out OUT      # train on every official row, save adapter + heads
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument('--fold', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--model', default=''); ap.add_argument('--data', default='/home/rohan/kagglecomp/name-that-llm/train.csv')
ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--head-lr', type=float, default=1e-3)
ap.add_argument('--prompts-per-batch', type=int, default=2); ap.add_argument('--max-len', type=int, default=384)
ap.add_argument('--rank', type=int, default=16); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(*x): print(f'[{(time.time() - T0) / 60:6.1f}m]', *x, flush=True)
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = 'cuda'

d = pd.read_csv(a.data); d['text'] = d.text.fillna('').astype(str)
classes = sorted(d.model.unique()); y = d.model.map({c: i for i, c in enumerate(classes)}).to_numpy(); groups = d.prompt_id.to_numpy(); texts = d.text.tolist()
if a.fold == 'all': tr, te = np.arange(len(d)), None
else: tr, te = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))[int(a.fold)]
log(f'train rows {len(tr)}, held-out rows {0 if te is None else len(te)}')

path = a.model
tok = AutoTokenizer.from_pretrained(path)
base = AutoModel.from_pretrained(path).to(torch.bfloat16)
n_layers = base.config.num_hidden_layers; hidden = base.config.hidden_size
depths = sorted({max(1, round(n_layers * f)) for f in (1 / 7, 0.25, 0.36, 0.5, 0.64, 0.79, 0.89, 1.0)})
base.config.use_cache = False; base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias='none',
                                        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])).to(dev)
norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in depths).to(dev); plain = nn.Linear(hidden * len(depths), 18).to(dev); cent = nn.Linear(hidden * len(depths), 18).to(dev)
drop = nn.Dropout(0.1)
log(f'{n_layers} layers, hidden {hidden}, pooling depths {depths}, LoRA params {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')


def features(batch_texts):
    enc = tok(batch_texts, padding=True, truncation=True, max_length=a.max_len, return_tensors='pt').to(dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hs = model(**enc, output_hidden_states=True).hidden_states
    m = enc['attention_mask'].unsqueeze(-1).float()
    return torch.cat([n((hs[L].float() * m).sum(1) / m.sum(1)) for L, n in zip(depths, norms)], -1)


def center(f, gid):
    n = int(gid.max()) + 1
    s = torch.zeros(n, f.shape[1], device=f.device).index_add_(0, gid, f); c = torch.bincount(gid, minlength=n).unsqueeze(1).float()
    return f - (s / c)[gid]


lora_params = [p for p in model.parameters() if p.requires_grad]
head_params = list(norms.parameters()) + list(plain.parameters()) + list(cent.parameters())
opt = torch.optim.AdamW([{'params': lora_params, 'lr': a.lr, 'weight_decay': 0.0}, {'params': head_params, 'lr': a.head_lr, 'weight_decay': 0.01}])
peak = [g['lr'] for g in opt.param_groups]
tr_prompts = np.unique(groups[tr]); rows_of = {p: tr[groups[tr] == p] for p in tr_prompts}
steps_per_epoch = math.ceil(len(tr_prompts) / a.prompts_per_batch); total = steps_per_epoch * a.epochs; warm = max(1, int(0.05 * total)); step = 0

model.train(); norms.train(); plain.train(); cent.train()
for epoch in range(a.epochs):
    order = rng.permutation(tr_prompts); losses = []
    for i in range(0, len(order), a.prompts_per_batch):
        rows = np.concatenate([rows_of[p] for p in order[i:i + a.prompts_per_batch]])
        gid = torch.as_tensor(np.unique(groups[rows], return_inverse=True)[1], device=dev); target = torch.as_tensor(y[rows], device=dev)
        f = features([texts[r] for r in rows])
        loss = F.cross_entropy(plain(drop(f)), target) + F.cross_entropy(cent(drop(center(f, gid))), target)
        loss.backward(); nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
        mult = (step + 1) / warm if step < warm else 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g, p in zip(opt.param_groups, peak): g['lr'] = p * mult
        opt.step(); opt.zero_grad(set_to_none=True); step += 1; losses.append(loss.item())
        if step % 10 == 0: log(f'epoch {epoch + 1}/{a.epochs} step {step}/{total} loss {np.mean(losses[-10:]):.3f} | GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
    log(f'epoch {epoch + 1} done, mean loss {np.mean(losses):.3f}')

model.save_pretrained(out / 'adapter')
torch.save({'norms': norms.state_dict(), 'plain': plain.state_dict(), 'cent': cent.state_dict(), 'depths': depths}, out / 'heads.pt')
if te is not None:
    model.eval(); norms.eval(); plain.eval(); cent.eval(); F_te = np.zeros((len(te), hidden * len(depths)), np.float32)
    with torch.no_grad():
        order = np.argsort([len(texts[r]) for r in te])
        for s in range(0, len(te), 32):
            idx = order[s:s + 32]; F_te[idx] = features([texts[te[j]] for j in idx]).cpu().numpy()
    Ft = torch.as_tensor(F_te, device=dev); gid = torch.as_tensor(np.unique(groups[te], return_inverse=True)[1], device=dev)
    with torch.no_grad():
        lp = F.log_softmax(plain(Ft), -1).cpu().numpy(); lc = F.log_softmax(cent(center(Ft, gid)), -1).cpu().numpy()
    mr = lambda p: float(np.mean([(p[y[te] == c] == c).mean() for c in np.unique(y[te])]))
    res = {'plain': mr(lp.argmax(1)), 'centred_true_groups': mr(lc.argmax(1)), 'mix_0.5': mr((0.5 * lp + 0.5 * lc).argmax(1)), 'sum': mr((lp + lc).argmax(1))}
    np.savez(out / 'heldout.npz', rows=te, feats=F_te.astype(np.float16), plain=lp, centred=lc)
    json.dump({**res, 'epochs': a.epochs, 'lr': a.lr, 'minutes': (time.time() - T0) / 60}, open(out / 'result.json', 'w'), indent=1)
    log('HELD-OUT FOLD', a.fold, json.dumps(res))
log('done')
