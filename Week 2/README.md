# Week 2

## Part 2 - MLP Language Model
Key learning:
- Implemented the Bengio et al. (2003) approach: map characters to learned embedding vectors, concatenate a fixed context window, pass through a hidden `tanh` layer, then a softmax output.
- Learned the train / dev (validation) / test split workflow and why you should never tune hyperparameters against the test set.
- Used a learning-rate sweep (`lre = torch.linspace(-3, 0, 1000)`) to find a good learning rate before committing to a full training run.
- Visualized the learned 2D character embeddings to see which characters the model treats as "similar."

## Part 3 - BatchNorm
Key learning:
- Understood why deep MLPs are hard to initialize well: saturated `tanh` units kill gradients, and small/large logit scales distort the initial loss.
- Implemented `BatchNorm1d` from scratch, including the train-mode batch statistics vs. eval-mode running statistics distinction.
- Used activation histograms, gradient histograms, and the gradient:data update-ratio plot as diagnostic tools to check whether every layer is learning at a healthy rate.

## MLP Exercises (`mlp.py`, E01-E03)
Key learning:
- Tuned hyperparameters (context length, embedding size, hidden size, learning-rate schedule) to try to beat the lecture's baseline validation loss.
- Quantified the effect of output-layer initialization: starting with large random output weights gives a loss far worse than `log(27)` (uniform-guessing baseline), while scaling the last layer down (`*0.01`, zero bias) starts training right at the uniform baseline.
- Implemented and benchmarked the Bengio et al. 2003 "direct connection" idea — a linear shortcut from the input embeddings straight to the output logits, bypassing the hidden `tanh` layer.

## BatchNorm Internals (`batchnorm.py`, E01-E02)
Key learning:
- Showed concretely why all-zero initialization is broken: with `W1 = b1 = W2 = 0`, every hidden neuron computes the identical value, so gradients are non-zero but identical across neurons — symmetry is never broken and the network only ever learns a rank-1 subspace.
- Derived the algebra for folding a BatchNorm layer into its preceding `Linear` layer at inference time (`W_new = (γ/σ)·W`, `b_new = β − γ·μ/σ`), and implemented it to produce a BatchNorm-free model with numerically identical outputs.
