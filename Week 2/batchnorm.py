"""
Week 2 Assignment – Part 2: BatchNorm Internals
================================================
E01 – Zero-initialization: what trains, what doesn't, and why
E02 – Fold BatchNorm gamma/beta into the preceding Linear layer

Run:  python part2_batchnorm.py
"""

import torch
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────────────────────────────────────
# Shared data setup
# ─────────────────────────────────────────────────────────────────────────────
words = open("names.txt").read().splitlines()
chars = sorted(set("".join(words)))
stoi  = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos  = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)    # 27

import random
random.seed(42)
random.shuffle(words)

BLOCK = 3
EMBD  = 10

def build_dataset(words, block_size=BLOCK):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr, Ytr   = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])


# ─────────────────────────────────────────────────────────────────────────────
# E01 – ALL WEIGHTS AND BIASES INITIALIZED TO ZERO
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("E01 – Zero Initialization Analysis")
print("=" * 60)

HIDDEN = 100

torch.manual_seed(0)
C_zero  = torch.zeros((vocab_size, EMBD),       requires_grad=True)
W1_zero = torch.zeros((BLOCK * EMBD, HIDDEN),   requires_grad=True)
b1_zero = torch.zeros(HIDDEN,                   requires_grad=True)
W2_zero = torch.zeros((HIDDEN, vocab_size),     requires_grad=True)
b2_zero = torch.zeros(vocab_size,               requires_grad=True)

params_zero = [C_zero, W1_zero, b1_zero, W2_zero, b2_zero]

print("\nInitial state (all zeros):")
emb_init    = C_zero[Xtr[:5]].view(5, -1)
print(f"  emb unique values  : {emb_init.unique().tolist()}")
pre_act     = emb_init @ W1_zero + b1_zero
print(f"  pre-tanh unique    : {pre_act.unique().tolist()}")
h_init      = torch.tanh(pre_act)
print(f"  h (post-tanh) uniq : {h_init.unique().tolist()}")
logits_init = h_init @ W2_zero + b2_zero
print(f"  logits unique      : {logits_init.unique().tolist()}")

with torch.no_grad():
    emb    = C_zero[Xtr[:64]].view(64, -1)
    h      = torch.tanh(emb @ W1_zero + b1_zero)
    logits = h @ W2_zero + b2_zero
    init_loss = F.cross_entropy(logits, Ytr[:64])
print(f"\nInitial loss = {init_loss.item():.4f}  (expected ≈ log(27) = {math.log(27):.4f})")

# Train a few steps and inspect
STEPS_ZERO = 5_000
losses_zero = []
for i in range(STEPS_ZERO):
    ix     = torch.randint(0, Xtr.shape[0], (64,))
    emb    = C_zero[Xtr[ix]].view(64, -1)
    h      = torch.tanh(emb @ W1_zero + b1_zero)
    logits = h @ W2_zero + b2_zero
    loss   = F.cross_entropy(logits, Ytr[ix])

    for p in params_zero: p.grad = None
    loss.backward()
    for p in params_zero: p.data += -0.1 * p.grad
    losses_zero.append(loss.item())

print(f"\nAfter {STEPS_ZERO} steps with lr=0.1:")
print(f"  Final loss = {losses_zero[-1]:.4f}")

# Inspect gradients and activations at step 0 (before any update)
torch.manual_seed(0)
C_z  = torch.zeros((vocab_size, EMBD),       requires_grad=True)
W1_z = torch.zeros((BLOCK * EMBD, HIDDEN),   requires_grad=True)
b1_z = torch.zeros(HIDDEN,                   requires_grad=True)
W2_z = torch.zeros((HIDDEN, vocab_size),     requires_grad=True)
b2_z = torch.zeros(vocab_size,               requires_grad=True)

ix  = torch.arange(64) % Xtr.shape[0]
emb = C_z[Xtr[ix]].view(64, -1)
h   = torch.tanh(emb @ W1_z + b1_z)
logits = h @ W2_z + b2_z
loss = F.cross_entropy(logits, Ytr[ix])
loss.backward()

print("\nGradient analysis at step 0:")
print(f"  C.grad  – mean={C_z.grad.abs().mean():.6f}, std={C_z.grad.std():.6f}")
print(f"  W1.grad – mean={W1_z.grad.abs().mean():.6f}, std={W1_z.grad.std():.6f}")
print(f"  b1.grad – mean={b1_z.grad.abs().mean():.6f}, std={b1_z.grad.std():.6f}")
print(f"  W2.grad – mean={W2_z.grad.abs().mean():.6f}, std={W2_z.grad.std():.6f}")
print(f"  b2.grad – mean={b2_z.grad.abs().mean():.6f}, std={b2_z.grad.std():.6f}")

# Check hidden layer: all neurons fire identically → dead neuron problem
print(f"\nHidden layer analysis:")
print(f"  h[0] (first sample, first 5 neurons): {h[0, :5].detach().tolist()}")
print(f"  h std across neurons = {h.std(dim=1).mean().item():.6f}")
print(f"  → all neurons are IDENTICAL → symmetry not broken")
print(f"  → W1 gradients: all rows of W1.grad identical?",
      (W1_z.grad[0] == W1_z.grad[1]).all().item())

print(f"""
EXPLANATION – Why the network only partially trains:
─────────────────────────────────────────────────────
The key issue is *symmetry*.

When W1 = 0 and b1 = 0:
  • pre_tanh = emb @ W1 + b1 = 0  for every neuron, every sample
  • tanh(0) = 0, so h = 0 for ALL neurons
  • logits = h @ W2 + b2 = 0 (since W2=0 too)

During backprop:
  • dL/dW2 = h^T @ dlogits = 0 @ ... = 0  → W2 gets NO gradient from h
  • dL/db2 gets gradient (it's added directly to logits) ✓
  • dL/db1 gets gradient (tanh'(0) = 1, so gradient flows) ✓
  • dL/dW1 = emb^T @ (dh) where dh = dL/dh * tanh'(pre) = ... * 1
    → W1 DOES get gradients, but because all neurons are identical,
    all rows/cols of W1.grad are the same → neurons never diversify!
  • dL/dC (embeddings): also gets gradient through W1

What trains:
  ✓ b2 (direct output bias) – has gradient, updates freely
  ✓ b1 (hidden bias) – has gradient since tanh'(0) = 1
  △ W1, W2 – get gradients but all neurons are symmetric clones
  △ C – gets gradients but diversity is limited

What stays broken:
  ✗ The hidden neurons never learn different features because symmetry
    is never broken. Every neuron is a perfect clone. The network
    effectively has only 1 neuron's worth of capacity regardless of
    HIDDEN size.

Bottom line: b2 learns (→ loss decreases somewhat from log(27)),
but W1 and W2 learn only a rank-1 subspace. Final loss is much worse
than a properly initialized network of the same size.
""")


# ─────────────────────────────────────────────────────────────────────────────
# E02 – FOLD BATCHNORM INTO PRECEDING LINEAR LAYER
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("E02 – Fold BatchNorm γ/β into Linear Layer")
print("=" * 60)

# ── Build a small 3-layer MLP with BatchNorm ─────────────────────────────────
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out) / fan_in**0.5
        self.bias   = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps      = eps
        self.momentum = momentum
        self.training = True
        # learnable params
        self.gamma = torch.ones(dim)
        self.beta  = torch.zeros(dim)
        # running stats (used at inference)
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(dim=0, keepdim=True)
            xvar  = x.var(dim=0,  keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar  = self.running_var
        xhat     = (x - xmean) / (xvar + self.eps).sqrt()
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar.squeeze()
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self): return []


# 3-layer MLP: Embed → Linear → BN → Tanh → Linear → BN → Tanh → Linear → output
torch.manual_seed(42)

H1, H2 = 64, 64

embed  = Linear(BLOCK * EMBD, H1, bias=False)   # no bias: BN absorbs it
bn1    = BatchNorm1d(H1)
act1   = Tanh()
lin2   = Linear(H1, H2, bias=False)
bn2    = BatchNorm1d(H2)
act2   = Tanh()
lin3   = Linear(H2, vocab_size, bias=True)       # last layer keeps its bias

C_mlp  = torch.randn((vocab_size, EMBD))

layers = [embed, bn1, act1, lin2, bn2, act2, lin3]
params_mlp = [C_mlp] + [p for l in layers for p in l.parameters()]
for p in params_mlp: p.requires_grad_(True)

print(f"Model parameters: {sum(p.nelement() for p in params_mlp):,}")

# ── Train ─────────────────────────────────────────────────────────────────────
STEPS_BN = 10_000
for i in range(STEPS_BN):
    ix  = torch.randint(0, Xtr.shape[0], (64,))
    emb = C_mlp[Xtr[ix]].view(64, -1)

    out = emb
    for l in layers:
        out = l(out)
    loss = F.cross_entropy(out, Ytr[ix])

    for p in params_mlp: p.grad = None
    loss.backward()
    lr = 0.1 if i < 7_000 else 0.01
    for p in params_mlp: p.data += -lr * p.grad

if i % 1_000 == 999:
    print(f"  step {i+1} loss={loss.item():.4f}")

# Set to eval mode (uses running stats)
for l in layers:
    if hasattr(l, 'training'):
        l.training = False

@torch.no_grad()
def forward_mlp(X, use_layers):
    """Forward pass through the 3-layer MLP."""
    emb = C_mlp[X].view(X.shape[0], -1)
    out = emb
    for l in use_layers:
        out = l(out)
    return out

with torch.no_grad():
    logits_before = forward_mlp(Xdev, layers)
    loss_before   = F.cross_entropy(logits_before, Ydev).item()

print(f"\nVal loss before folding : {loss_before:.6f}")

# ── Fold BN into Linear ───────────────────────────────────────────────────────
# For a Linear layer followed by BatchNorm:
#   BN(Wx) = γ * (Wx - μ) / σ + β
#           = (γ/σ) * W * x  +  (β - γ*μ/σ)
#
# So the folded layer has:
#   W_new = (γ/σ) * W          (scale each output feature)
#   b_new = β - γ * μ / σ

def fold_bn_into_linear(linear, bn):
    """
    Returns (W_new, b_new) that are equivalent to:
        y = BN(linear(x))   [at inference time]
    """
    with torch.no_grad():
        # BN running stats
        mu    = bn.running_mean           # (H,)
        var   = bn.running_var            # (H,)
        gamma = bn.gamma                  # (H,)
        beta  = bn.beta                   # (H,)
        eps   = bn.eps
        sigma = (var + eps).sqrt()        # (H,)

        # W shape: (fan_in, fan_out); scale each column (output feature)
        scale  = gamma / sigma            # (H,)
        W_new  = linear.weight * scale    # broadcast: (fan_in, H) * (H,)
        b_orig = linear.bias if linear.bias is not None else torch.zeros(linear.weight.shape[1])
        b_new  = beta + (b_orig - mu) * scale

    return W_new.detach(), b_new.detach()

W1_fold, b1_fold = fold_bn_into_linear(embed, bn1)
W2_fold, b2_fold = fold_bn_into_linear(lin2,  bn2)

print(f"\nFolding results:")
print(f"  embed.weight shape : {embed.weight.shape} → W1_fold: {W1_fold.shape}")
print(f"  lin2.weight  shape : {lin2.weight.shape}  → W2_fold: {W2_fold.shape}")

# ── Build the folded model (no BatchNorm layers) ───────────────────────────────
class FoldedLinear:
    def __init__(self, W, b):
        self.weight = W
        self.bias   = b
    def __call__(self, x):
        return x @ self.weight + self.bias

fl1  = FoldedLinear(W1_fold, b1_fold)
fl2  = FoldedLinear(W2_fold, b2_fold)
# lin3 has no BN after it, reuse as-is
layers_folded = [fl1, act1, fl2, act2, lin3]

@torch.no_grad()
def forward_folded(X):
    emb = C_mlp[X].view(X.shape[0], -1)
    out = emb
    for l in layers_folded:
        out = l(out)
    return out

logits_folded = forward_folded(Xdev)
loss_folded   = F.cross_entropy(logits_folded, Ydev).item()

print(f"\nVal loss after  folding : {loss_folded:.6f}")
print(f"  Difference            : {abs(loss_before - loss_folded):.2e}")

# Verify logits match numerically
max_diff = (logits_before - logits_folded).abs().max().item()
print(f"\nMax absolute logit difference (before vs folded): {max_diff:.2e}")

if max_diff < 1e-5:
    print("  ✓ VERIFIED: folded model produces identical outputs (within floating point)")
else:
    print("  ✗ Warning: outputs differ by more than expected.")

print(f"""
Mathematical derivation:
─────────────────────────────────────────────────────
At inference, BatchNorm computes:
    BN(z) = γ * (z - μ_run) / sqrt(σ²_run + ε) + β

Where z = Wx  (output of linear layer, ignoring bias for simplicity).

So:
    BN(Wx) = γ/σ · Wx  +  β - γ·μ/σ
           = W_new · x  +  b_new

where:
    W_new = diag(γ/σ) @ W    (scale rows of W)
    b_new = β - γ·μ/σ

This means BatchNorm at inference time is EXACTLY equivalent to
a plain linear layer with adjusted weights. The BN parameters
(γ, β, μ_run, σ²_run) are all absorbed into W_new and b_new.

Benefits of folding:
  • Fewer operations at inference (no mean/var computation)
  • No extra memory for BN running stats
  • Exactly equivalent: zero accuracy loss
""")

print("Part 2 complete!")
