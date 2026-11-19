# Week 2: Calculus, probability and backprop by hand

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuranLearning/ai-llm-deep-dive/blob/main/notebooks/week-02-backprop-by-hand.ipynb) · [Practice notebook](../notebooks/week-02-backprop-by-hand.ipynb) · [Learning Hub week page](https://claude.ai/artifact/VwTfCas4FMnB23QHuBUJhj#w-ai-llm-2)

**This week:** Watch: [3Blue1Brown · Neural Networks · eps. 1–4](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) · Read: [Mathematics for Machine Learning (free PDF) · ch. 5–6](https://mml-book.github.io/) · Build: Hand-derive backprop for a 2-layer MLP

---

Training a neural network is calculus plus probability: probability says what "good" means (the loss), and calculus says which way to move every weight to get better. This week ends with you deriving backpropagation for a 2-layer network by hand.

## 1. Derivatives as sensitivity

The derivative `df/dx` answers: if I nudge x by a tiny amount ε, how much does f move? Roughly `f(x + ε) ≈ f(x) + ε · df/dx`. For a function of many inputs, the **gradient** `∇f` collects all the partial derivatives into one vector. It points in the direction of steepest increase, so gradient descent steps the other way:

```
w ← w − η · ∇L(w)        # η is the learning rate
```

## 2. The chain rule is the whole of backprop

If `y = f(u)` and `u = g(x)`, then `dy/dx = (dy/du) · (du/dx)`. A network is a long chain of such functions. Backprop runs the chain rule from the loss backwards, reusing each intermediate result once, so computing all gradients costs about the same as two forward passes.

Think of it as a **computational graph**. Each node knows two things: how to compute its output (forward) and how to turn the gradient arriving at its output into gradients for its inputs (backward). Three local rules cover most of a network:

| Node | Forward | Backward (given upstream gradient g) |
| --- | --- | --- |
| add | `z = a + b` | `da = g`, `db = g` |
| multiply | `z = a · b` | `da = g · b`, `db = g · a` |
| tanh | `z = tanh(a)` | `da = g · (1 − z²)` |

When a value feeds several nodes, its gradients **add up**. That single rule explains why micrograd uses `+=` in its backward pass (week 5).

## 3. Probability: what the loss means

A language model outputs a probability distribution over the next token. Given the true token y, the loss is the **negative log-likelihood**:

```
L = −log p(y)
```

If the model gives the right token probability 0.9, the loss is 0.105; at 0.01 it is 4.6. Averaged over a dataset this is the **cross-entropy** between the data and the model. Its exponential is the **perplexity**: a perplexity of 20 means the model is, on average, as unsure as a uniform choice among 20 tokens.

Two facts from probability you will reuse:

- **Expectation**: `E[x] = Σ p(x) · x`. Training minimizes the expected loss over the data.
- **Variance of a sum** of independent terms adds up. This is why we scale attention scores (week 1) and why weight initialization uses `1/sqrt(fan_in)`.

## 4. Softmax + cross-entropy: the gradient worth memorizing

With logits `z`, probabilities `p = softmax(z)` and a one-hot target `y`, the gradient of `L = −Σ y_i log p_i` with respect to the logits is simply:

```
dL/dz = p − y
```

The model's prediction minus the truth. Every classifier and every language model starts its backward pass here.

## 5. Backprop for a 2-layer MLP

Forward pass for one example `x` (shape `(d,)`), hidden size h, C classes:

```
a1 = W1 x + b1          # (h,)
h1 = tanh(a1)           # (h,)
z  = W2 h1 + b2         # (C,)
p  = softmax(z)
L  = −log p[y]
```

Backward pass, applying the rules above from the top:

```
dz  = p − onehot(y)              # (C,)
dW2 = outer(dz, h1)              # (C, h)
db2 = dz
dh1 = W2ᵀ dz                     # (h,)
da1 = dh1 * (1 − h1**2)          # tanh'
dW1 = outer(da1, x)              # (h, d)
db1 = da1
```

Check the shapes: every gradient has the same shape as the thing it is the gradient of. For a batch, the outer products become matrix products and you average over the batch.

## 6. How to verify a gradient

Compare your analytic gradient with a numerical one:

```
dL/dw ≈ (L(w + ε) − L(w − ε)) / (2ε),   ε ≈ 1e-5
```

The relative error `|a − n| / max(|a|, |n|)` should be below about 1e-6 in float64. You will build this check next week.

## This week's build

On paper (then typed into a notebook), derive the backward pass above yourself without looking, including the batch version. Note one place where you got stuck; that note goes in your weekly journal below.

## Self-check

1. Why do gradients add when one value feeds two nodes?
2. A model assigns 0.25 to the correct token. What is the loss, and what is the perplexity if every token looks like this?
3. Derive `d/da tanh(a)` in terms of `tanh(a)`.
4. What is the shape of `dW1` for a batch of 32, d = 10, h = 64?
5. In your own words, why is `dL/dz = p − y` so convenient?
