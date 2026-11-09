"""Score passages with the public 'L24' LoRA experts (dhruvtibarewal/nlm-l24-adapters-v5: q14v5, q8v5, ll8v5), offline.

Recipe (read from the public notebooks, own implementation): the base model is TRUNCATED to its first `keep` decoder layers, a rank-16 LoRA
on every attention/MLP projection is applied, the hidden states at `layers` (indices into hidden_states, 0 = embeddings) are mean-pooled over
real tokens (first token masked), each goes through its own LayerNorm, and the concatenation feeds two linear heads: plain and prompt-centred.

  python l24_worker.py --base DIR --adapters A0,A1,... --query q.csv --out out.npz [--keep 24] [--layers 2,4,6,8,12,16,20,24]

query.csv has columns id,text. For every adapter the output holds the raw plain logits and the raw centred-head logits (centring needs prompt
groups, so it is applied by the caller: centred = raw - group_mean(raw) + bias). LoRA is applied as a separate fp32 low-rank path (no merging, so
no rounding drift between adapters). Results are written after every adapter.
"""
import argparse, gc, json, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

T0 = time.time()
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def log(*x): print(f'[{(time.time() - T0) / 60:6.1f}m]', *x, flush=True)


class LoRALinear(nn.Module):
    """Frozen linear layer plus an optional fp32 low-rank update. The factors live in a plain tuple (not registered), so model.to(dtype) never touches them."""
    def __init__(self, base):
        super().__init__(); self.base_layer = base; self.ab = None; self.scale = 1.0

    def forward(self, x):
        out = self.base_layer(x)
        if self.ab is None: return out
        a, b = self.ab
        return (out + (x.float() @ a.t() @ b.t()) * self.scale).to(out.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--adapters', required=True); ap.add_argument('--query', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--keep', type=int, default=24); ap.add_argument('--layers', default='2,4,6,8,12,16,20,24')
    ap.add_argument('--max-len', type=int, default=384); ap.add_argument('--bs', type=int, default=16)
    a = ap.parse_args()
    adapters = a.adapters.split(','); layers = [int(x) for x in a.layers.split(',')]
    frame = pd.read_csv(a.query, dtype={'id': str}); texts = frame.text.fillna('').astype(str).tolist()
    if not len(texts) or not frame.id.is_unique: raise SystemExit('query must have unique ids and at least one row')

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.base); tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    full_layers = tc.num_hidden_layers; tc.num_hidden_layers = a.keep; tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None: tc.layer_types = tc.layer_types[:a.keep]
    # memory budget per GPU: 13 GiB on small cards (Kaggle T4, 15 GB: identical to the Kaggle runs), 85% of the card on big ones (A100, RTX 6000) so a 14B model stays on one GPU
    budget = {i: f'{13 if torch.cuda.get_device_properties(i).total_memory / 2**30 < 20 else int(0.85 * torch.cuda.get_device_properties(i).total_memory / 2**30)}GiB' for i in range(torch.cuda.device_count())}
    model = AutoModel.from_pretrained(a.base, config=cfg, dtype=torch.float16, device_map='auto', max_memory=budget).eval()
    first = next(model.parameters()).device
    wrappers = {}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and name.rsplit('.', 1)[-1] in TARGETS and re.match(r'layers\.\d+\.', name):
            parent, attr = name.rsplit('.', 1); w = LoRALinear(mod); setattr(model.get_submodule(parent), attr, w); wrappers[name] = w
    if len(wrappers) != 7 * a.keep: raise SystemExit(f'expected {7 * a.keep} LoRA targets, found {len(wrappers)}')
    log(f'base loaded: kept {a.keep} of {full_layers} layers, {len(wrappers)} LoRA targets, devices {sorted({str(w.base_layer.weight.device) for w in wrappers.values()})}')

    enc = tok(texts, truncation=True, max_length=a.max_len)['input_ids']
    order = np.argsort([len(x) for x in enc], kind='stable'); N = len(enc)
    log(f'{N} passages, mean {np.mean([len(x) for x in enc]):.0f} tokens, longest {max(len(x) for x in enc)}')

    @torch.no_grad()
    def forward_batch(b):
        Lm = max(len(enc[j]) for j in b)
        ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)
        for r, j in enumerate(b): ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
        ids, att = ids.to(first), att.to(first)
        hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
        redo = False
        if not all(bool(torch.isfinite(hs[l]).all()) for l in layers):   # fp16 overflow in un-normed deep states: redo this batch in bf16
            del hs; model.to(torch.bfloat16); redo = True
            hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
        mask = att.unsqueeze(-1).float(); mask[:, 0] = 0
        pooled = [((hs[l].float() * mask.to(hs[l].device)).sum(1) / mask.sum(1).to(hs[l].device)).to(first) for l in layers]
        del hs
        if redo: model.to(torch.float16)
        return pooled, redo

    res = {'plain': [], 'centred_raw': [], 'centred_bias': []}; timing = []; overflow = 0
    for ai, ad in enumerate(adapters):
        t = time.time(); ad = Path(ad)
        cfg_a = json.loads((ad / 'adapter_config.json').read_text()); scale = cfg_a['lora_alpha'] / cfg_a['r']
        sd = load_file(str(ad / 'adapter_model.safetensors')); used = skipped = 0
        for w in wrappers.values(): w.ab = None
        for k, A in sd.items():
            if '.lora_A.' not in k: continue
            m = re.search(r'layers\.(\d+)\.((?:self_attn|mlp)\.\w+)\.lora_A\.', k)
            if not m: raise SystemExit(f'unexpected adapter key {k}')
            if int(m.group(1)) >= a.keep: skipped += 1; continue
            w = wrappers[f'layers.{m.group(1)}.{m.group(2)}']; B = sd[k.replace('.lora_A.', '.lora_B.')]
            dev = w.base_layer.weight.device; w.ab = (A.float().to(dev), B.float().to(dev)); w.scale = scale; used += 1
        if used != len(wrappers): raise SystemExit(f'{ad.name}: adapter covers {used} of {len(wrappers)} targets ({skipped} skipped beyond layer {a.keep})')
        H = torch.load(ad / 'heads.pt', map_location='cpu', weights_only=True)
        hid = H['norms.0.weight'].shape[0]; norms = []
        for i in range(len(layers)):
            n = nn.LayerNorm(hid); n.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']}); norms.append(n.to(first).float())
        Wp, bp = H['plain.weight'].float().to(first), H['plain.bias'].float().to(first)
        Wc, bc = H['centered.weight'].float().to(first), H['centered.bias'].float().to(first)
        plain = np.zeros((N, 18), np.float32); cent = np.zeros((N, 18), np.float32)
        for s in range(0, N, a.bs):
            b = order[s:s + a.bs]; pooled, redo = forward_batch(b); overflow += int(redo)
            with torch.no_grad():
                Z = torch.cat([n(p) for n, p in zip(norms, pooled)], -1)
                plain[b] = (Z @ Wp.t() + bp).cpu().numpy(); cent[b] = (Z @ Wc.t() + bc).cpu().numpy()
            if (s // a.bs) % 40 == 0: log(f'  {ad.name}: {s}/{N}')
        if not (np.isfinite(plain).all() and np.isfinite(cent).all()): raise SystemExit(f'{ad.name}: non-finite scores')
        res['plain'].append(plain); res['centred_raw'].append(cent); res['centred_bias'].append(bc.cpu().numpy())
        timing.append({'adapter': ad.name, 'seconds': time.time() - t}); del sd, H; gc.collect()
        log(f'{ad.name} done in {(time.time() - t) / 60:.1f} min (bf16 redo batches so far: {overflow})')
        np.savez(a.out, ids=frame.id.to_numpy(dtype=str), adapters=np.asarray([Path(x).name for x in adapters[:ai + 1]]),
                 plain=np.stack(res['plain']), centred_raw=np.stack(res['centred_raw']), centred_bias=np.stack(res['centred_bias']))
    Path(a.out).with_suffix('.json').write_text(json.dumps({'passages': N, 'keep': a.keep, 'layers': layers, 'adapter_seconds': timing, 'bf16_redo_batches': overflow,
                                                           'total_seconds': time.time() - T0, 'peak_gb': [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())]}, indent=1))
    log('done')


if __name__ == '__main__': main()
