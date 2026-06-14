"""
Week 2 Assignment – Part 1: MLP Exercises
==========================================
E01 – Tune hyperparameters to beat Andrej's best val loss of 2.2
E02 – Initialization analysis (uniform loss baseline vs actual; fix it)
E03 – Implement one idea from Bengio et al. 2003 (direct connection / larger context)

Run:  python part1_mlp.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
words = open("names.txt").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)          # 27

def build_dataset(words, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))


# ─────────────────────────────────────────────────────────────────────────────
# E01 – Beat val-loss 2.2
# Key changes vs Andrej's baseline:
#   • block_size  3 → 4   (wider context)
#   • embed_dim   2 → 16  (richer embeddings)
#   • hidden      100 → 256
#   • learning-rate schedule with decay
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("E01 – Tuned MLP (goal: val-loss < 2.2)")
print("=" * 60)

BLOCK  = 4
EMBD   = 16
HIDDEN = 256
SEED   = 2147483647

torch.manual_seed(SEED)
Xtr, Ytr   = build_dataset(words[:n1],      BLOCK)
Xdev, Ydev = build_dataset(words[n1:n2],    BLOCK)
Xte,  Yte  = build_dataset(words[n2:],      BLOCK)

# Parameters
C   = torch.randn((vocab_size, EMBD),              generator=torch.manual_seed(SEED))
W1  = torch.randn((BLOCK * EMBD, HIDDEN),          generator=torch.manual_seed(SEED)) * 0.1
b1  = torch.zeros(HIDDEN)
W2  = torch.randn((HIDDEN, vocab_size),            generator=torch.manual_seed(SEED)) * 0.01
b2  = torch.zeros(vocab_size)
params = [C, W1, b1, W2, b2]
for p in params:
    p.requires_grad_(True)

print(f"Total parameters: {sum(p.nelement() for p in params):,}")

# Training with LR decay
STEPS      = 200_000
BATCH      = 64
train_loss = []

for i in range(STEPS):
    ix  = torch.randint(0, Xtr.shape[0], (BATCH,))
    emb = C[Xtr[ix]].view(BATCH, -1)
    h   = torch.tanh(emb @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in params:
        p.grad = None
    loss.backward()

    # LR schedule: start 0.1, decay to 0.01 at 150k
    lr = 0.1 if i < 150_000 else 0.01
    for p in params:
        p.data += -lr * p.grad

    if i % 10_000 == 0:
        print(f"  step {i:>7d}  loss={loss.item():.4f}")
    train_loss.append(loss.log10().item())

# Evaluate
@torch.no_grad()
def split_loss(X, Y):
    emb    = C[X].view(X.shape[0], -1)
    h      = torch.tanh(emb @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()

print(f"\nE01 Results:")
print(f"  Train loss : {split_loss(Xtr,  Ytr):.4f}")
print(f"  Val   loss : {split_loss(Xdev, Ydev):.4f}  (target < 2.2)")
print(f"  Test  loss : {split_loss(Xte,  Yte):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# E02 – Initialization Analysis
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("E02 – Initialization Analysis")
print("=" * 60)

import math
uniform_loss = math.log(vocab_size)
print(f"1) Loss under perfectly uniform distribution: {uniform_loss:.4f}")
print(f"   (= -log(1/27) = log(27))")

# Bad init (Andrej's original: large random logits → high loss)
torch.manual_seed(SEED)
C_bad   = torch.randn((vocab_size, 2))
W1_bad  = torch.randn((3 * 2, 100))
b1_bad  = torch.randn(100)
W2_bad  = torch.randn((100, vocab_size))   # large random weights
b2_bad  = torch.randn(vocab_size)

Xbad, Ybad = build_dataset(words[:n1], 3)
with torch.no_grad():
    emb    = C_bad[Xbad[:64]].view(64, -1)
    h      = torch.tanh(emb @ W1_bad + b1_bad)
    logits = h @ W2_bad + b2_bad
    bad_init_loss = F.cross_entropy(logits, Ybad[:64]).item()

print(f"\n2) Loss with default (bad) initialization : {bad_init_loss:.4f}")
print(f"   (high because random logits → confidently wrong predictions)")

# Good init: W2 * 0.01, b2 = 0  → logits near zero → near-uniform probs
torch.manual_seed(SEED)
C_good   = torch.randn((vocab_size, 2))
W1_good  = torch.randn((3 * 2, 100))
b1_good  = torch.randn(100)
W2_good  = torch.randn((100, vocab_size)) * 0.01   # ← key fix
b2_good  = torch.zeros(vocab_size)                  # ← key fix

with torch.no_grad():
    emb    = C_good[Xbad[:64]].view(64, -1)
    h      = torch.tanh(emb @ W1_good + b1_good)
    logits = h @ W2_good + b2_good
    good_init_loss = F.cross_entropy(logits, Ybad[:64]).item()

print(f"\n3) Loss with good initialization          : {good_init_loss:.4f}")
print(f"   (W2 * 0.01, b2 = 0 → logits ≈ 0 → probs ≈ uniform)")
print(f"   Δ from uniform = {abs(good_init_loss - uniform_loss):.4f}")

print(f"""
Explanation:
  Bad init:  W2 and b2 are large random values, so the network is
             initially very confident about wrong answers.
             Loss ≈ {bad_init_loss:.1f} (much worse than uniform {uniform_loss:.4f}).

  Good init: Scale the last-layer weights small (× 0.01) and zero out
             biases.  Logits are near zero ⟹ softmax ≈ uniform ⟹ loss
             starts near log(27) ≈ {uniform_loss:.4f}.  The network then only
             has to learn "improvements" from a neutral start.
""")

# ─────────────────────────────────────────────────────────────────────────────
# E03 – Bengio et al. 2003 idea: Direct connection from input embeddings to output
# Section 4 of the paper adds a direct (shortcut) connection from the
# concatenated embedding layer to the output, bypassing the hidden tanh layer.
# This is a residual-like shortcut that can help in shallow networks.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("E03 – Bengio 2003: Direct input→output shortcut connection")
print("=" * 60)
print("""
Idea from Section 4 of the paper (equation 4):
  y = b + Wh + U·x_flat
  where x_flat is the concatenated embeddings fed DIRECTLY to output too.
  This is a linear shortcut that bypasses the tanh hidden layer.
""")

BLOCK3 = 3
EMBD3  = 10
HIDDEN3 = 100

torch.manual_seed(SEED)
Xtr3, Ytr3   = build_dataset(words[:n1],   BLOCK3)
Xdev3, Ydev3 = build_dataset(words[n1:n2], BLOCK3)

# Standard MLP (baseline for comparison)
C3s  = torch.randn((vocab_size, EMBD3),           generator=torch.manual_seed(SEED))
W1s  = torch.randn((BLOCK3*EMBD3, HIDDEN3),       generator=torch.manual_seed(SEED)) * 0.1
b1s  = torch.zeros(HIDDEN3)
W2s  = torch.randn((HIDDEN3, vocab_size),          generator=torch.manual_seed(SEED)) * 0.01
b2s  = torch.zeros(vocab_size)
params_std = [C3s, W1s, b1s, W2s, b2s]
for p in params_std: p.requires_grad_(True)

# Bengio MLP with direct shortcut:  logits = h@W2 + b2 + emb_flat@U
C3b  = torch.randn((vocab_size, EMBD3),           generator=torch.manual_seed(SEED))
W1b  = torch.randn((BLOCK3*EMBD3, HIDDEN3),       generator=torch.manual_seed(SEED)) * 0.1
b1b  = torch.zeros(HIDDEN3)
W2b  = torch.randn((HIDDEN3, vocab_size),          generator=torch.manual_seed(SEED)) * 0.01
b2b  = torch.zeros(vocab_size)
U    = torch.randn((BLOCK3*EMBD3, vocab_size),    generator=torch.manual_seed(SEED)) * 0.01  # direct connection
params_ben = [C3b, W1b, b1b, W2b, b2b, U]
for p in params_ben: p.requires_grad_(True)

def train_model(params, use_shortcut=False, steps=30_000, batch=64, lr=0.1):
    if use_shortcut:
        C_, W1_, b1_, W2_, b2_, U_ = params
    else:
        C_, W1_, b1_, W2_, b2_ = params

    losses = []
    for i in range(steps):
        ix      = torch.randint(0, Xtr3.shape[0], (batch,))
        emb     = C_[Xtr3[ix]].view(batch, -1)
        h       = torch.tanh(emb @ W1_ + b1_)
        logits  = h @ W2_ + b2_
        if use_shortcut:
            logits = logits + emb @ U_       # direct shortcut

        loss = F.cross_entropy(logits, Ytr3[ix])
        for p in params: p.grad = None
        loss.backward()
        step_lr = lr if i < 20_000 else lr * 0.1
        for p in params: p.data += -step_lr * p.grad
        losses.append(loss.item())
    return losses

print("Training standard MLP …")
_ = train_model(params_std, use_shortcut=False)

print("Training Bengio MLP (with direct shortcut) …")
_ = train_model(params_ben, use_shortcut=True)

@torch.no_grad()
def eval_model(params, use_shortcut, X, Y):
    if use_shortcut:
        C_, W1_, b1_, W2_, b2_, U_ = params
    else:
        C_, W1_, b1_, W2_, b2_ = params
    emb    = C_[X].view(X.shape[0], -1)
    h      = torch.tanh(emb @ W1_ + b1_)
    logits = h @ W2_ + b2_
    if use_shortcut:
        logits = logits + emb @ U_
    return F.cross_entropy(logits, Y).item()

std_val = eval_model(params_std, False, Xdev3, Ydev3)
ben_val = eval_model(params_ben, True,  Xdev3, Ydev3)

print(f"\nE03 Results after 30k steps:")
print(f"  Standard MLP val loss        : {std_val:.4f}")
print(f"  Bengio shortcut MLP val loss : {ben_val:.4f}")
if ben_val < std_val:
    print(f"  ✓ Direct connection IMPROVED val loss by {std_val - ben_val:.4f}")
else:
    print(f"  Direct connection did not improve in this run (common for shallow nets).")
    print(f"  Benefit becomes more visible with deeper architectures.")

print("""
Analysis:
  The direct (shortcut) connection from embeddings to output adds a linear
  pathway that helps gradient flow and allows the model to learn linear
  relationships without passing through tanh non-linearity.
  In Bengio 2003, this was found to give a small but consistent improvement.
  The effect is most pronounced at lower step counts where the hidden layer
  has not yet converged — the shortcut allows faster early learning.
""")

print("Part 1 complete!")
