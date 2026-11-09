"""Score passages with the trained Gemma LoRA classifier, offline. Mirrors the scoring path of worker_unsloth.py step for step:
same model load, same LoRA layout, same decoder, same three pooled layers, same head. Writes log-probabilities to --out.

  python score_gemma.py --check                                   # only import the stack and print versions
  python score_gemma.py --base DIR --adapter DIR --head head.pt --texts texts.parquet --out out.npz
"""
import argparse, importlib.metadata, importlib.util, sys, time
from pathlib import Path


def _torchao_too_old():
    try:
        version = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        return False
    from packaging.version import parse
    return parse(version) < parse('0.16.0')


if _torchao_too_old():   # peft refuses old torchao builds. Nothing here uses torchao, so hide it before peft looks.
    _find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *args, **kw: None if name.split('.')[0] == 'torchao' else _find_spec(name, *args, **kw)

ap = argparse.ArgumentParser()
ap.add_argument('--check', action='store_true')
ap.add_argument('--base'); ap.add_argument('--adapter'); ap.add_argument('--head'); ap.add_argument('--texts'); ap.add_argument('--out')
ap.add_argument('--n-depths', type=int, default=3); ap.add_argument('--max-len', type=int, default=384); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--rank', type=int, default=16)
a = ap.parse_args()

import unsloth  # noqa: F401  (must load before transformers and peft)
from unsloth import FastModel
import numpy as np, pandas as pd, torch
from torch import nn

if a.check:
    import bitsandbytes, peft, transformers
    print('unsloth', unsloth.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__,
          '| bitsandbytes', bitsandbytes.__version__, '| gpus', torch.cuda.device_count())
    sys.exit(0)

T0 = time.time()
def log(*x): print(f'[{(time.time() - T0) / 60:6.1f}m]', *x, flush=True)

dev = 'cuda'; compute = torch.float16    # the training run used the default, fp16 (Unsloth keeps Gemma's activations in fp32 itself)
texts = pd.read_parquet(a.texts).text.fillna('').astype(str).tolist()

model, processor = FastModel.from_pretrained(model_name=a.base, max_seq_length=a.max_len, dtype=compute, load_in_4bit=True,
                                             full_finetuning=False, use_gradient_checkpointing='unsloth', random_state=a.seed)
tok = getattr(processor, 'tokenizer', processor)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = 'right'
model = FastModel.get_peft_model(model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0, bias='none', random_state=a.seed,
                                 finetune_vision_layers=False, finetune_language_layers=True,
                                 finetune_attention_modules=True, finetune_mlp_modules=True, use_gradient_checkpointing='unsloth')
log('loaded base model and LoRA layout')


def find_decoder(m):
    best = None
    for name, mod in m.named_modules():
        if any(w in name.lower() for w in ('vision', 'audio', 'visual')): continue
        layers = getattr(mod, 'layers', None)
        if isinstance(layers, nn.ModuleList) and hasattr(mod, 'embed_tokens') and (best is None or len(layers) > len(best[1].layers)):
            best = (name, mod)
    if best is None: raise SystemExit('could not find the text decoder inside the model')
    return best
dec_name, decoder = find_decoder(model)
n_layers = len(decoder.layers)
picks = sorted({max(1, round(n_layers * f)) for f in (0.11, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)}) if a.n_depths == 8 else sorted({n_layers // 2, 3 * n_layers // 4, n_layers})   # must match the training run
hidden = int(getattr(getattr(model.config, 'text_config', model.config), 'hidden_size'))
log('decoder', dec_name or '(root)', n_layers, 'layers, pooling', picks)


class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in picks); self.out = nn.Linear(hidden * len(picks), 18)
    def features(self, pooled): return torch.cat([n(p) for n, p in zip(self.norms, pooled)], -1)
    def forward(self, pooled): return self.out(self.features(pooled))
head = Head().to(dev)

# same order as training: for_training first, then adapters to fp32, so the saved fp32 weights load without rounding
FastModel.for_training(model)
for p in model.parameters():
    if p.requires_grad and p.dtype != torch.float32: p.data = p.data.float()

import json
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
cfg = json.loads((Path(a.adapter) / 'adapter_config.json').read_text())
if int(cfg.get('r', a.rank)) != a.rank: raise SystemExit(f'adapter rank {cfg.get("r")} differs from --rank {a.rank}')
state = load_file(str(Path(a.adapter) / 'adapter_model.safetensors'))
result = set_peft_model_state_dict(model, state)
unexpected = list(getattr(result, 'unexpected_keys', None) or [])
if unexpected: raise SystemExit(f'{len(unexpected)} adapter tensors did not match the model, first: {unexpected[:3]}')
lora_b = sum(float(p.detach().float().abs().sum()) for n, p in model.named_parameters() if 'lora_B' in n)
if not lora_b > 0: raise SystemExit('every LoRA B matrix is still zero after loading: the adapter was not applied')
head.load_state_dict(torch.load(a.head, map_location='cpu'))
log(f'adapter loaded ({len(state)} tensors, |B| sum {lora_b:.1f}) and head loaded')

enc = tok(texts, truncation=True, max_length=a.max_len)['input_ids']
def batch(rows): return tok.pad({'input_ids': [enc[int(i)] for i in rows]}, return_tensors='pt').to(dev)
def pool_layers(b):
    with torch.autocast(dev, dtype=compute):
        hs = decoder(input_ids=b['input_ids'], attention_mask=b['attention_mask'], output_hidden_states=True, use_cache=False).hidden_states
    if hs is None or len(hs) != n_layers + 1: raise SystemExit(f'the decoder returned {None if hs is None else len(hs)} hidden states, expected {n_layers + 1}')
    m = b['attention_mask'].clone().float(); m[:, 0] = 0; m = m.unsqueeze(-1); den = m.sum(1).clamp(min=1)
    return [(hs[l].float() * m).sum(1) / den for l in picks]

model.eval(); head.eval()
N = len(enc); logp = np.zeros((N, 18), np.float32)
with torch.inference_mode():
    idx = np.argsort([len(x) for x in enc], kind='stable')
    for s in range(0, N, 16):
        rows = idx[s:s + 16]
        logp[rows] = torch.log_softmax(head.out(head.features(pool_layers(batch(rows)))), -1).cpu().numpy()
        if (s // 16) % 50 == 0: log(f'scored {s}/{N}')
if not np.isfinite(logp).all(): raise SystemExit('non-finite predictions')
np.savez(a.out, logp=logp)
log(f'done: {N} passages scored, saved to {a.out}')
