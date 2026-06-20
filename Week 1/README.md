# Week 1

## Part 0 - Micrograd: Building an Autograd Engine
Key learning:
- Built a scalar-valued `Value` class from scratch that records a computation graph (`_prev`, `_op`) as operations are applied.
- Implemented reverse-mode autodiff: topological sort of the graph followed by calling each node's local `_backward()` in reverse order.
- Built `Neuron`, `Layer`, and `MLP` classes on top of `Value` and trained a tiny network with manual gradient descent (`p.data += -lr * p.grad`) on a 4-example toy dataset.
- Visualized the computation graph with graphviz to see how gradients flow backward through `+`, `*`, `tanh`, etc.
- Cross-checked gradients against PyTorch to confirm the from-scratch engine is correct.

## Micrograd Exercises
Key learning:
- Compared three ways of getting a derivative: analytical (calculus by hand), one-sided numerical finite difference, and the symmetric derivative — and confirmed the symmetric version is far more accurate at the same step size `h`.
- Extended `Value` with the operators it was missing (`__neg__`, `__sub__`, `__mul__`, `__pow__`, `__truediv__`, `exp`, `log`) in order to support softmax and negative log-likelihood loss, not just squared error.
- Verified the hand-built engine's gradients exactly match PyTorch's autograd on a softmax + NLL example.

## Part 1 - Bigram Language Model
Key learning:
- Built the simplest possible character-level language model: counting how often each character follows another, normalized into a probability table.
- Learned why count-based models need smoothing (add-one/Laplace) to avoid zero probabilities for unseen bigrams.
- Re-derived the exact same counting model as a one-layer neural net trained with gradient descent — showing that "counting" and "softmax + cross-entropy" are the same thing in this case.
- Practiced sampling new names from a learned probability distribution and evaluating a model with average negative log-likelihood.
