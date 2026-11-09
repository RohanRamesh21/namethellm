"""Train one Unsloth QLoRA classifier on the pool rows, then score the official rows, which it never trains on.
Saves a checkpoint every few minutes and resumes from it, so a crash costs minutes, not the whole run."""
import argparse, importlib.metadata, importlib.util, json, math, os, resource, time
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
ap.add_argument('--name', required=True); ap.add_argument('--repo', required=True); ap.add_argument('--dtype', choices=['fp16', 'bf16'], default='fp16')
ap.add_argument('--seed', type=int, required=True); ap.add_argument('--data', required=True); ap.add_argument('--work', required=True)
ap.add_argument('--budget-min', type=float, default=120); ap.add_argument('--deadline', type=float, default=0)
ap.add_argument('--bs', type=int, default=8); ap.add_argument('--accum', type=int, default=4); ap.add_argument('--max-len', type=int, default=384)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--n-depths', type=int, default=3); ap.add_argument('--init-adapter', default=''); ap.add_argument('--init-head', default=''); ap.add_argument('--train-file', default=''); ap.add_argument('--score-file', default=''); ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--shard-index', type=int, default=0); ap.add_argument('--shard-count', type=int, default=1)
ap.add_argument('--limit-steps', type=int, default=0); ap.add_argument('--eval-rows', type=int, default=0); ap.add_argument('--ckpt-every', type=int, default=100)
a = ap.parse_args()

import unsloth  # noqa: F401  (must load before transformers and peft)
from unsloth import FastModel
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn import functional as F

T0 = time.time()
def log(*x): print(f'[{(time.time() - T0) / 60:7.1f}m {a.name}]', *x, flush=True)
def memory():
    free = None
    try:
        free = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable')) / 2**20
    except (OSError, StopIteration):
        pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    gpu = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    return f'RAM free {free:.1f}GB, process peak {rss:.1f}GB, GPU peak {gpu:.1f}GB' if free is not None else f'process peak {rss:.1f}GB, GPU peak {gpu:.1f}GB'

out = Path(a.work) / 'runs' / a.name; out.mkdir(parents=True, exist_ok=True)
ckpt_path = out / 'ckpt.pt'
status = json.loads((out / 'run.json').read_text()) if (out / 'run.json').exists() else {}
status.update(name=a.name, repo=a.repo, seed=a.seed, status='running')
def save_status(**kw):
    status.update(kw); tmp = out / 'run.json.tmp'; tmp.write_text(json.dumps(status, indent=1)); os.replace(tmp, out / 'run.json')
def fail(reason):
    save_status(status='failed', reason=reason); raise SystemExit(f'{a.name}: {reason}')
save_status()

torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = 'cuda'
compute = torch.float16 if a.dtype == 'fp16' else torch.bfloat16
train = pd.read_parquet(a.train_file if a.train_file else Path(a.work) / 'train_pool.parquet')   # stage 2 trains on a file of official rows
if a.shard_count > 1:   # this run trains on its own slice: every model keeps the same share, so the slices do not overlap
    train = train[train.groupby('model').cumcount() % a.shard_count == a.shard_index].reset_index(drop=True)
official = pd.read_csv(Path(a.data) / 'train.csv'); official['text'] = official.text.fillna('').astype(str)
labels = sorted(official.model.unique()); to_id = {l: i for i, l in enumerate(labels)}
if a.score_file: official = pd.read_parquet(a.score_file)[['text', 'model']].reset_index(drop=True); official['text'] = official.text.fillna('').astype(str)
if a.eval_rows: official = official.head(a.eval_rows)
y = train.model.map(to_id).to_numpy()

model, processor = FastModel.from_pretrained(model_name=a.repo, max_seq_length=a.max_len, dtype=compute, load_in_4bit=True,
                                             full_finetuning=False, use_gradient_checkpointing='unsloth', random_state=a.seed)
tok = getattr(processor, 'tokenizer', processor)   # multimodal models return a processor
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = 'right'
model = FastModel.get_peft_model(model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0, bias='none', random_state=a.seed,
                                 finetune_vision_layers=False, finetune_language_layers=True,
                                 finetune_attention_modules=True, finetune_mlp_modules=True, use_gradient_checkpointing='unsloth')
log('loaded', memory())

def find_decoder(m):
    """The text decoder stack: the largest module with .layers and .embed_tokens outside any vision or audio tower."""
    best = None
    for name, mod in m.named_modules():
        if any(w in name.lower() for w in ('vision', 'audio', 'visual')): continue
        layers = getattr(mod, 'layers', None)
        if isinstance(layers, nn.ModuleList) and hasattr(mod, 'embed_tokens') and (best is None or len(layers) > len(best[1].layers)):
            best = (name, mod)
    if best is None: fail('could not find the text decoder inside the model')
    return best
dec_name, decoder = find_decoder(model)
n_layers = len(decoder.layers)
picks = sorted({max(1, round(n_layers * f)) for f in (0.11, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)}) if a.n_depths == 8 else sorted({n_layers // 2, 3 * n_layers // 4, n_layers})   # 8 depths as in the public embedding expert
hidden = int(getattr(getattr(model.config, 'text_config', model.config), 'hidden_size'))
log('decoder', dec_name or '(root)', n_layers, 'layers, pooling', picks)

class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in picks); self.out = nn.Linear(hidden * len(picks), 18)
    def features(self, pooled): return torch.cat([n(p) for n, p in zip(self.norms, pooled)], -1)
    def forward(self, pooled): return self.out(self.features(pooled))
head = Head().to(dev)

enc = tok(train.text.tolist(), truncation=True, max_length=a.max_len)['input_ids']
enc_off = tok(official.text.tolist(), truncation=True, max_length=a.max_len)['input_ids']
lens = np.array([len(x) for x in enc]); log('rows', len(enc), 'mean tokens', round(float(lens.mean())))

def batch(rows, ids):
    return tok.pad({'input_ids': [ids[int(i)] for i in rows]}, return_tensors='pt').to(dev)
def pool_layers(b):
    """Mean over real tokens of a few layers, skipping position 0. The layers come from the decoder's own hidden_states:
    reading them with forward hooks would capture tensors that Unsloth's checkpointing detaches, so LoRA would get no gradient."""
    with torch.autocast(dev, dtype=compute):
        hs = decoder(input_ids=b['input_ids'], attention_mask=b['attention_mask'], output_hidden_states=True, use_cache=False).hidden_states
    if hs is None or len(hs) != n_layers + 1: fail(f'the decoder returned {None if hs is None else len(hs)} hidden states, expected {n_layers + 1}')
    m = b['attention_mask'].clone().float(); m[:, 0] = 0; m = m.unsqueeze(-1); den = m.sum(1).clamp(min=1)
    return [(hs[l].float() * m).sum(1) / den for l in picks]

model.eval()
with torch.inference_mode():   # fail in seconds, not hours, if this precision overflows on real passages
    probe = pool_layers(batch(np.argsort(-lens)[:8], enc))
if not all(torch.isfinite(p).all() for p in probe): fail(f'non-finite activations in {a.dtype}')
FastModel.for_training(model); head.train()

named = [(f'lora.{k}', p) for k, p in model.named_parameters() if p.requires_grad] + [(f'head.{k}', p) for k, p in head.named_parameters()]
for _, p in named:   # adapters must be fp32 for loss scaling to work
    if p.dtype != torch.float32: p.data = p.data.float()
if a.init_adapter:   # stage 2 starts from the stage-1 adapter and head (adapters are already fp32 here, so the saved weights load exactly)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    res = set_peft_model_state_dict(model, load_file(str(Path(a.init_adapter) / 'adapter_model.safetensors')))
    unexpected = list(getattr(res, 'unexpected_keys', None) or [])
    if unexpected: fail(f'{len(unexpected)} adapter tensors did not match the model, first: {unexpected[:3]}')
    head.load_state_dict(torch.load(a.init_head, map_location='cpu'))
    log('initialised from the stage-1 adapter and head')
lora = [p for k, p in named if k.startswith('lora.')]
log('trainable', sum(p.numel() for p in lora), 'LoRA tensors', len(lora))
opt = torch.optim.AdamW([{'params': lora, 'lr': a.lr, 'weight_decay': 0.0}, {'params': list(head.parameters()), 'lr': a.lr * 10, 'weight_decay': 0.01}])
peak = [g['lr'] for g in opt.param_groups]
scaler = torch.amp.GradScaler('cuda', enabled=a.dtype == 'fp16')

order = np.concatenate([rng.permutation(len(enc)) for _ in range(a.epochs)]); micro = []
for s in range(0, len(order), a.bs * 64):   # group similar lengths so little compute goes on padding
    chunk = order[s:s + a.bs * 64]; chunk = chunk[np.argsort(lens[chunk], kind='stable')]
    micro += [chunk[i:i + a.bs] for i in range(0, len(chunk), a.bs)]
micro = [micro[i] for i in rng.permutation(len(micro))]
total = math.ceil(len(micro) / a.accum)
if a.limit_steps: total = min(total, a.limit_steps)
step = seen = start_micro = 0; per = None; trained_sec = 0.0; recent = []; resumed = False
start_norms = [float(p.detach().float().norm()) for p in lora]

if ckpt_path.exists():   # resume where the last attempt stopped
    st = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    params = dict(named)
    for k, t in st['trainable'].items(): params[k].data.copy_(t.to(dev))
    opt.load_state_dict(st['opt']); scaler.load_state_dict(st['scaler'])
    step, seen, start_micro, total, per, trained_sec, recent = st['step'], st['seen'], st['micro'], st['total'], st['per'], st['trained_sec'], st['recent']
    log(f'resumed at step {step} of {total} ({seen} rows done)'); save_status(resumed_at_step=step); resumed = True

def save_ckpt(next_micro):
    st = {'trainable': {k: p.detach().cpu() for k, p in named}, 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': step, 'seen': seen,
          'micro': next_micro, 'total': total, 'per': per, 'trained_sec': trained_sec, 'recent': recent[-200:]}
    tmp = out / 'ckpt.pt.tmp'; torch.save(st, tmp); os.replace(tmp, ckpt_path)

def budget_left():
    left = a.budget_min * 60 - trained_sec
    if a.deadline: left = min(left, a.deadline - time.time() - 600)   # keep 10 minutes for scoring and the report
    return left
if budget_left() <= 0 and step < total: fail('no time left')
if per and step < total:   # a resumed run re-fits its schedule to the time that is left
    total = min(total, step + max(1, int(budget_left() / per)))

def set_lr():
    warm = max(1, int(0.03 * total))
    mult = (step + 1) / warm if step < warm else 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    for g, p in zip(opt.param_groups, peak): g['lr'] = p * mult

TEST_CRASH = int(os.environ.get('NTL_TEST_CRASH_AT_STEP', '0'))   # used only by the local tests
checked_grad = False; t_last = time.time(); t_measure = None; loss_sum = 0.0; bad = 0
for i in range(start_micro, len(micro)):
    if step >= total: break
    rows = micro[i]; b = batch(rows, enc); target = torch.as_tensor(y[rows], device=dev)
    loss = F.cross_entropy(head(pool_layers(b)), target)
    if not torch.isfinite(loss):
        bad += 1
        if bad >= 10: fail('loss went non-finite')
        continue
    scaler.scale(loss / a.accum).backward(); seen += len(rows); loss_sum += loss.item()
    if not checked_grad:   # the silent failure mode: the head learns but LoRA never gets a gradient
        # count NaN or inf as present: loss scaling makes early fp16 gradients overflow, and the scaler then skips that step
        reached = sum(1 for p in lora if p.grad is not None and bool(p.grad.detach().ne(0).any()))
        finite = all(bool(torch.isfinite(p.grad).all()) for p in lora if p.grad is not None)
        if reached == 0: fail('LoRA received no gradient: the hidden states are detached from the adapters')
        log(f'gradient check passed: {reached} of {len(lora)} LoRA tensors reached ({"finite" if finite else "overflowed, the scaler will adjust"})'); checked_grad = True
    if (i + 1) % a.accum: continue
    scaler.unscale_(opt); nn.utils.clip_grad_norm_([p for _, p in named], 1.0)
    set_lr(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
    recent.append(loss_sum / a.accum); loss_sum = 0.0
    now = time.time(); trained_sec += now - t_last; t_last = now
    if step == min(10, total) and not resumed:   # the adapters must actually move, not just receive gradients that every step skips
        changed = sum(1 for p, n0 in zip(lora, start_norms) if abs(float(p.detach().float().norm()) - n0) > 1e-7)
        if changed == 0: fail(f'LoRA weights unchanged after {step} steps (loss scale {scaler.get_scale():.0f}): the adapters are not training')
        log(f'training check passed: {changed} of {len(lora)} LoRA tensors changed in {step} steps')
    if step == 3 and per is None: t_measure = now; s_measure = step
    if per is None and t_measure and step == s_measure + 20 and not a.limit_steps:   # fit the schedule to the budget so the learning rate still decays
        per = (now - t_measure) / 20; fit = step + int(budget_left() / per)
        if fit < total: log(f'budget allows {fit} of {total} steps'); total = fit
        log(f'{per:.2f}s per step; about {(total - step) * per / 60:.0f} minutes left to train')
    if step % 50 == 0: log(f'step {step}/{total} rows {seen} loss {np.mean(recent[-50:]):.3f} scale {scaler.get_scale():.0f} | {memory()}'); save_status(step=step, planned_steps=total, rows_seen=seen)
    if step % a.ckpt_every == 0: save_ckpt(i + 1)
    if TEST_CRASH and step == TEST_CRASH: log('test crash'); os._exit(137)
save_ckpt(len(micro))   # a crash while scoring then resumes straight to scoring
train_min = trained_sec / 60

model.eval(); head.eval(); N = len(enc_off); logp = np.zeros((N, 18), np.float32); feats = None
with torch.inference_mode():
    idx = np.argsort([len(x) for x in enc_off], kind='stable')
    for s in range(0, N, 16):
        rows = idx[s:s + 16]; f = head.features(pool_layers(batch(rows, enc_off)))
        logp[rows] = torch.log_softmax(head.out(f), -1).cpu().numpy()
        if feats is None: feats = np.zeros((N, f.shape[1]), np.float16)
        feats[rows] = f.clamp(-60000, 60000).cpu().numpy()
if not np.isfinite(logp).all(): fail('non-finite predictions')
np.savez(out / 'pred.npz', logp=logp, feats=feats)
model.save_pretrained(out / 'adapter'); torch.save(head.state_dict(), out / 'head.pt')
true = official.model.map(to_id).to_numpy(); pred = logp.argmax(1)
score = float(np.mean([(pred[true == k] == k).mean() for k in range(18) if (true == k).any()]))
save_status(status='done', step=step, planned_steps=total, rows_seen=seen, train_minutes=round(train_min, 1),
            final_loss=round(float(np.mean(recent[-50:])), 4) if recent else None, official_score=round(score, 4), peak=memory())
ckpt_path.unlink(missing_ok=True)
log('done', json.dumps(status))
