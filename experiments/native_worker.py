"""Score passages with the public 'native embedding expert' (CHEN Han, Qwen3-Embedding-8B + rank-16 LoRA + 8 pooled depths),
offline, one fold adapter at a time. Needs two T4s: the fp16 backbone is split over both.

  python native_worker.py --base MODELDIR --assets ASSETDIR --query query.csv --out out.npz --work WORKDIR [--folds 0,1,2,3,4]

query.csv has columns id,text. The output holds, for every fold, the plain and centred head logits (centring is applied later,
because it needs prompt groups) and the centred head's bias. A fold's results are saved as soon as it finishes.
"""
import argparse, gc, json, os, time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

INSTRUCTION = 'Instruct: Identify the language model that generated the text based on its writing style.\nQuery: '
DEPTHS = (3, 6, 9, 12, 18, 24, 30, 36)
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
MAX_LEN = 512
END_OF_TEXT = 151643
T0 = time.time()


def log(*x): print(f'[{(time.time() - T0) / 60:6.1f}m]', *x, flush=True)


class TaskLoRA(nn.Module):
    """Frozen linear layer plus a rank-16 update kept in fp32 (alpha 32 / rank 16 = scale 2)."""
    def __init__(self, base):
        super().__init__()
        self.base_layer = base
        self.a = nn.Linear(base.in_features, 16, bias=False, device=base.weight.device, dtype=torch.float32)
        self.b = nn.Linear(16, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)

    def forward(self, x):
        out = self.base_layer(x)
        return (out + self.b(self.a(x.float())) * 2.0).to(out.dtype)


class Expert:
    def __init__(self, base, assets, work):
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        for key in ('allow_tf32', 'allow_fp16_reduced_precision_reduction', 'allow_bf16_reduced_precision_reduction'):
            setattr(torch.backends.cuda.matmul, key, False)
        # Two GPUs (Kaggle's 2 x T4 15 GB): the fp16 backbone (16.4 GB) is split over both. One GPU: it must hold the whole backbone plus activations (24 GB or more).
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu < 1: raise SystemExit('no GPU found')
        if self.n_gpu == 1 and torch.cuda.get_device_properties(0).total_memory / 2**30 < 22:
            raise SystemExit('a single GPU needs at least 22 GB for the 8B backbone in fp16; use two GPUs or a larger card')
        self.assets = Path(assets); base = Path(base)
        self.labels = json.loads((self.assets / 'manifest.json').read_text())['classes']
        index = json.loads((base / 'model.safetensors.index.json').read_text())
        shards = sorted(set(index['weight_map'].values()))
        # configuration and tokenizer come from the asset folder (the training copies); the weight shards are linked from the model input
        stage = Path(work) / 'native-base'; stage.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'):
            src = self.assets / 'base' / name
            if not src.is_file(): raise SystemExit(f'missing {src}')
            (stage / name).write_bytes(src.read_bytes())
        for name in shards:
            if not (base / name).is_file(): raise SystemExit(f'missing weight shard {base / name}')
            link = stage / name
            if not link.exists(): link.symlink_to(base / name)
        cfg = json.loads((stage / 'config.json').read_text())
        if cfg['num_hidden_layers'] != 36 or cfg['hidden_size'] != 4096: raise SystemExit('unexpected backbone configuration')
        self.tok = AutoTokenizer.from_pretrained(stage, local_files_only=True); self.tok.padding_side = 'right'
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        last = 1 if self.n_gpu >= 2 else 0   # with two GPUs: layers 0-15 on GPU 0, layers 16-35 and the final norm on GPU 1 (unchanged from the Kaggle runs)
        placement = {'embed_tokens': 0, 'norm': last, 'rotary_emb': last}
        placement.update({f'layers.{i}': 0 if i < 16 or last == 0 else 1 for i in range(36)})
        self.model, info = AutoModel.from_pretrained(stage, local_files_only=True, torch_dtype=torch.float16, device_map=placement,
                                                     attn_implementation='sdpa', output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs', 'unexpected_keys')): raise SystemExit(f'backbone did not load cleanly: {info}')
        self.model.config.use_cache = False
        self.lora = {}
        for name, layer in list(self.model.named_modules()):
            if isinstance(layer, nn.Linear) and name.rsplit('.', 1)[-1] in TARGETS:
                wrapped = TaskLoRA(layer); parent, attr = name.rsplit('.', 1)
                setattr(self.model.get_submodule(parent), attr, wrapped); self.lora[name] = wrapped
        if len(self.lora) != 252: raise SystemExit(f'expected 252 LoRA targets, found {len(self.lora)}')
        self.norms = nn.ModuleList(nn.LayerNorm(4096, device='cuda:0', dtype=torch.float32) for _ in DEPTHS)
        self.plain = nn.Linear(4096 * len(DEPTHS), 18, device='cuda:0', dtype=torch.float32)
        self.centred = nn.Linear(4096 * len(DEPTHS), 18, device='cuda:0', dtype=torch.float32)
        self.model.requires_grad_(False).eval()
        for m in (self.norms, self.plain, self.centred): m.requires_grad_(False).eval()
        self.slots = {}
        for name, m in self.lora.items():
            self.slots[f'encoder.backbone.base_model.model.{name}.lora_A.default.weight'] = m.a.weight
            self.slots[f'encoder.backbone.base_model.model.{name}.lora_B.default.weight'] = m.b.weight
        for prefix, m in (('norms', self.norms), ('plain', self.plain), ('centered', self.centred)):
            self.slots.update({f'{prefix}.{n}': p for n, p in m.named_parameters()})
        if len(self.slots) != 524: raise SystemExit(f'expected 524 trainable tensors, found {len(self.slots)}')
        self.current = None
        log('backbone and LoRA layout ready; GPU memory', [round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())], 'GB')

    @torch.inference_mode()
    def load_fold(self, fold):
        state = load_file(str(self.assets / f'fold{fold}' / 'trainables.safetensors'))
        if set(state) != set(self.slots): raise SystemExit(f'fold {fold}: adapter tensors do not match: {sorted(set(state) ^ set(self.slots))[:4]}')
        for name, p in self.slots.items():
            t = state[name]
            if t.shape != p.shape or t.dtype != torch.float32 or not torch.isfinite(t).all(): raise SystemExit(f'fold {fold}: bad tensor {name}')
            p.copy_(t.to(p.device))
        del state; gc.collect(); self.current = fold

    def windows(self, texts):
        """Each passage becomes covering windows of at most 512 tokens (the instruction repeated on every later window); no tail is dropped."""
        prefix = self.tok(INSTRUCTION, add_special_tokens=False)['input_ids']
        if self.tok('', add_special_tokens=True)['input_ids'] != [END_OF_TEXT]: raise SystemExit('the tokenizer does not append the end-of-text token')
        capacity = MAX_LEN - len(prefix) - 1; first = MAX_LEN - 1
        full = self.tok([INSTRUCTION + str(t) for t in texts], truncation=False, add_special_tokens=False)['input_ids']
        wins, owner, weight = [], [], []
        for row, toks in enumerate(full):
            wins.append(toks[:first] + [END_OF_TEXT]); owner.append(row); weight.append(max(1, min(len(toks), first) - len(prefix)))
            pos = first
            while pos < len(toks):
                part = toks[pos:pos + capacity]; wins.append(prefix + part + [END_OF_TEXT]); owner.append(row); weight.append(len(part)); pos += len(part)
        return wins, np.asarray(owner), np.asarray(weight, np.float64)

    @torch.inference_mode()
    def run_fold(self, wins, owner, weight, n_texts, batch_size=4):
        order = sorted(range(len(wins)), key=lambda i: (len(wins[i]), wins[i]))
        acc = {k: np.zeros((n_texts, 18)) for k in ('plain', 'centred')}
        for s in range(0, len(order), batch_size):
            pick = order[s:s + batch_size]
            b = self.tok.pad({'input_ids': [wins[i] for i in pick]}, padding=True, return_tensors='pt').to('cuda:0')
            out = self.model(**b, output_hidden_states=True, use_cache=False, return_dict=True)
            if len(out.hidden_states) != 37: raise SystemExit('unexpected number of hidden states')
            last = b['attention_mask'].sum(1) - 1
            parts = []
            for depth, norm in zip(DEPTHS, self.norms):
                h = out.last_hidden_state if depth == 36 else out.hidden_states[depth]
                v = h[torch.arange(len(pick), device=h.device), last.to(h.device)].float().to('cuda:0')
                if depth == 36: v = F.normalize(v, dim=-1)
                parts.append(norm(v))
            feats = torch.cat(parts, -1)
            for key, head in (('plain', self.plain), ('centred', self.centred)):
                np.add.at(acc[key], owner[pick], head(feats).cpu().numpy().astype(np.float64) * weight[pick, None])
            del out, feats, parts, b
            if (s // batch_size) % 100 == 0: log(f'  fold {self.current}: {s}/{len(order)} windows')
        den = np.bincount(owner, weights=weight, minlength=n_texts)[:, None]
        return acc['plain'] / den, acc['centred'] / den


def main():
    ap = argparse.ArgumentParser()
    for name in ('base', 'assets', 'query', 'out', 'work'): ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--folds', default='0,1,2,3,4'); ap.add_argument('--batch-size', type=int, default=4)
    a = ap.parse_args()
    frame = pd.read_csv(a.query, dtype={'id': str}); texts = frame.text.fillna('').astype(str).tolist()
    if not len(texts) or not frame.id.is_unique: raise SystemExit('query must have unique ids and at least one row')
    folds = [int(f) for f in a.folds.split(',')]
    expert = Expert(a.base, a.assets, a.work)
    wins, owner, weight = expert.windows(texts)
    log(f'{len(texts)} passages -> {len(wins)} windows, longest {max(len(w) for w in wins)} tokens, mean {np.mean([len(w) for w in wins]):.0f}')
    res = {'plain': [], 'centred': [], 'bias': []}; timing = []
    for fold in folds:
        t = time.time(); expert.load_fold(fold)
        p, c = expert.run_fold(wins, owner, weight, len(texts), a.batch_size)
        if not (np.isfinite(p).all() and np.isfinite(c).all()): raise SystemExit(f'fold {fold}: non-finite scores')
        res['plain'].append(p); res['centred'].append(c); res['bias'].append(expert.centred.bias.detach().cpu().numpy().copy())
        timing.append({'fold': fold, 'seconds': time.time() - t})
        log(f'fold {fold} done in {(time.time() - t) / 60:.1f} min')
        np.savez(a.out, ids=frame.id.to_numpy(dtype=str), labels=np.asarray(expert.labels), folds=np.asarray(folds[:len(res['plain'])]),
                 plain=np.stack(res['plain']), centred=np.stack(res['centred']), centred_bias=np.stack(res['bias']))
    a.out.with_suffix('.json').write_text(json.dumps({'passages': len(texts), 'windows': len(wins), 'fold_seconds': timing, 'total_seconds': time.time() - T0,
                                                     'peak_gb': [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())]}, indent=1))
    log('done')


if __name__ == '__main__': main()
