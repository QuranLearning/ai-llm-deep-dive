# Week 3: NumPy MLP and the Adam optimizer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuranLearning/ai-llm-deep-dive/blob/main/notebooks/week-03-numpy-mlp-adam.ipynb) · [Practice notebook](../notebooks/week-03-numpy-mlp-adam.ipynb) · [Learning Hub week page](https://claude.ai/artifact/VwTfCas4FMnB23QHuBUJhj#w-ai-llm-3)

**This week:** Read: [Adam (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980) · Build: Logistic regression + MLP in pure NumPy with manual gradients

---

This is a coding week. You turn last week's derivation into a working NumPy model, prove the gradients are right, and learn the optimizer that trains nearly every LLM: Adam.

## 1. The plan

1. Logistic regression on a 2-class toy dataset (two Gaussian blobs).
2. A 2-layer MLP on a non-linear dataset (two interleaved spirals or "moons").
3. A gradient check for both.
4. Plain SGD first, then Adam, and compare the loss curves.

Keep everything in float64 while you debug; switch to float32 only once the gradient check passes.

## 2. Logistic regression in ten lines

```python
import numpy as np
def sigmoid(z): return 1 / (1 + np.exp(-z))

def loss_and_grads(w, b, X, y):          # X: (N, d), y: (N,) in {0,1}
    p = sigmoid(X @ w + b)               # (N,)
    eps = 1e-12
    L = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
    dz = (p - y) / len(y)                # same "p − y" as softmax
    return L, X.T @ dz, dz.sum()
```

Train with `w -= lr * dw; b -= lr * db` for a few hundred steps and plot the decision boundary. A straight line is all it can draw, so it will fail on spirals. That failure is the motivation for hidden layers.

## 3. The MLP

Use the forward and backward passes from week 2, in batch form:

```python
def forward(params, X):
    W1, b1, W2, b2 = params
    H = np.tanh(X @ W1 + b1)             # (N, h)
    Z = H @ W2 + b2                      # (N, C)
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return H, P

def backward(params, X, y, H, P):
    W1, b1, W2, b2 = params
    N = len(y)
    dZ = P.copy(); dZ[np.arange(N), y] -= 1; dZ /= N
    dW2 = H.T @ dZ;  db2 = dZ.sum(0)
    dH  = dZ @ W2.T
    dA  = dH * (1 - H**2)
    dW1 = X.T @ dA;  db1 = dA.sum(0)
    return [dW1, db1, dW2, db2]
```

Initialize `W1` with `rng.normal(size=(d, h)) / np.sqrt(d)` and biases with zeros. Too-large initial weights saturate tanh and stall learning; you will meet this again in makemore part 3.

## 4. Gradient check

```python
def grad_check(f, params, grads, eps=1e-5, n=20):
    rng = np.random.default_rng(1)
    for P, G in zip(params, grads):
        for _ in range(n):
            i = tuple(rng.integers(s) for s in P.shape)
            old = P[i]
            P[i] = old + eps; Lp = f()
            P[i] = old - eps; Lm = f()
            P[i] = old
            num = (Lp - Lm) / (2*eps)
            rel = abs(num - G[i]) / max(1e-12, abs(num) + abs(G[i]))
            assert rel < 1e-5, (i, num, G[i])
```

If it fails, check one layer at a time. The usual culprits are a missing division by N, a wrong transpose, or forgetting that the max subtraction in softmax needs no gradient.

## 5. Adam, explained

SGD uses one learning rate for every weight. Adam keeps two running averages per weight and adapts the step size:

```
m ← β1·m + (1 − β1)·g          # momentum: average gradient
v ← β2·v + (1 − β2)·g²         # average squared gradient
m̂ = m / (1 − β1^t)             # bias correction for early steps
v̂ = v / (1 − β2^t)
w ← w − η · m̂ / (sqrt(v̂) + ε)
```

Typical values: `β1 = 0.9`, `β2 = 0.999` (LLMs often use 0.95), `ε = 1e-8`. Dividing by `sqrt(v̂)` gives weights with small, rare gradients relatively larger steps. **AdamW** adds weight decay directly to the weights (`w ← w − η·λ·w`) instead of through the gradient. That is the version used to train GPT-style models, and the one you will use in PyTorch next week.

## 6. What to look at

- Plot training loss for SGD and Adam on the same axes.
- Plot the decision boundary every 100 steps and watch it bend.
- Try h = 2, 8 and 64 hidden units. Note in your journal which is the smallest that solves the spirals.

## Reading: the Adam paper

Read sections 1–3 of Kingma & Ba (2014). Skip the convergence proof. Focus on Algorithm 1 and on the explanation of bias correction; you should be able to say why `m` starts biased towards zero.

## Self-check

1. Why can't logistic regression separate two spirals?
2. Your gradient check passes for `W2` but fails for `W1`. Name two likely causes.
3. What happens to Adam's step size for a weight whose gradient is always tiny?
4. Why is bias correction needed in the first few steps?
5. What is the difference between L2 regularization and AdamW's decoupled weight decay?
