# Week 1: Linear algebra for attention

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuranLearning/ai-llm-deep-dive/blob/main/notebooks/week-01-linear-algebra.ipynb) · [Practice notebook](../notebooks/week-01-linear-algebra.ipynb) · [Learning Hub week page](https://claude.ai/artifact/VwTfCas4FMnB23QHuBUJhj#w-ai-llm-1)

**This week:** Watch: [3Blue1Brown · Essence of Linear Algebra · eps. 1–8](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · Read: [Mathematics for Machine Learning (free PDF) · ch. 2](https://mml-book.github.io/) · Build: One-page note: the matrix ops inside attention

---

Every number inside a language model lives in a vector, and almost every operation is a matrix multiplication. This week you rebuild the intuition for both, aimed at one target: by Sunday you should be able to read the attention formula and say what every symbol does.

## 1. Vectors are points with meaning

A vector is a list of numbers, and you can picture it as an arrow from the origin. In an LLM each token becomes a vector called an **embedding**: GPT-2 small uses 768 numbers per token. The model learns these numbers so that tokens used in similar ways end up near each other.

Two operations matter most:

- **Addition and scaling.** `a + b` moves along both arrows; `2a` doubles the arrow's length. A residual connection in a transformer is literally `x + f(x)`.
- **The dot product.** `a · b = a1·b1 + a2·b2 + … + an·bn`. Geometrically it equals `|a| |b| cos θ`, so it is large when two vectors point the same way, zero when they are perpendicular and negative when they point apart.

> The dot product is the model's measure of "how relevant is this to that". Attention is built on it.

**Worked example.** Let `q = [1, 2]`, `k1 = [2, 1]`, `k2 = [-1, 0.5]`.
`q · k1 = 1·2 + 2·1 = 4` and `q · k2 = 1·(−1) + 2·0.5 = 0`. Key 1 matches the query; key 2 is unrelated.

## 2. A matrix is a batch of dot products

A matrix `W` with shape `(m, n)` maps an n-dimensional vector to an m-dimensional one. Each output number is the dot product of one row of `W` with the input. Read `y = W x` as "ask m different questions of x, one per row".

Matrix-matrix multiplication `C = A B` with `A: (m, n)` and `B: (n, p)` gives `C: (m, p)`, where `C[i, j]` is row i of A dotted with column j of B. The inner sizes must match; the outer sizes survive. Checking shapes this way catches most bugs in deep-learning code before you run anything.

Three facts you will use constantly:

1. `(A B)ᵀ = Bᵀ Aᵀ`. Transposes reverse the order.
2. A linear layer is `y = x W + b`. With a batch of B rows, `x` is `(B, n)` and `W` is `(n, m)`, giving `(B, m)`.
3. Matrix multiplication is not commutative: `A B ≠ B A` in general.

## 3. From dot products to attention

Stack T token vectors as rows of a matrix `X` with shape `(T, d)`. A transformer makes three projections of it:

```
Q = X W_q     # queries: what each token is looking for   (T, d_k)
K = X W_k     # keys:    what each token offers           (T, d_k)
V = X W_v     # values:  what each token passes along     (T, d_v)
```

Then one matrix product scores every token against every other token:

```
scores = Q Kᵀ / sqrt(d_k)        # (T, T)
weights = softmax(scores, row-wise)
out = weights V                  # (T, d_v)
```

Row i of `scores` holds the dot products between token i's query and every key. Softmax turns each row into positive weights that sum to 1, and `weights V` takes a weighted average of the value vectors. That is self-attention: each token builds a new vector by mixing in the tokens most relevant to it.

Why divide by `sqrt(d_k)`? If the entries of q and k are roughly independent with variance 1, their dot product has variance `d_k`. Large scores push softmax towards a one-hot output with tiny gradients. Dividing by `sqrt(d_k)` brings the variance back to about 1.

## 4. Softmax in one line

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

For `z = [4, 0]`, `exp(4) ≈ 54.6` and `exp(0) = 1`, so the weights are about `[0.982, 0.018]`. In code, always subtract the row maximum before `exp`; the result is unchanged and you avoid overflow.

## 5. Try it in NumPy

```python
import numpy as np
T, d = 4, 8
rng = np.random.default_rng(0)
X  = rng.normal(size=(T, d))
Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv
S = Q @ K.T / np.sqrt(d)
S -= S.max(axis=1, keepdims=True)
W = np.exp(S); W /= W.sum(axis=1, keepdims=True)
out = W @ V
print(W.round(2), W.sum(axis=1), out.shape)
```

Change one row of `X` and watch which weights move. Then add a causal mask (set `S[i, j] = -inf` for `j > i` before softmax) so each token only sees earlier ones. That mask is the only difference between GPT-style and BERT-style attention.

## This week's build

Write a one-page note, in your own words, that explains each line of the attention formula above with the shapes written next to it. You will reuse it in weeks 10–11 when you build GPT.

## Self-check

1. What does a negative dot product between a query and a key mean for attention?
2. `A` is `(32, 768)` and `W` is `(768, 3072)`. What is the shape of `A W`, and what layer of a transformer does it look like?
3. Why does the scores matrix have shape `(T, T)`, and what does entry `[i, j]` mean?
4. Compute `softmax([2, 2, 0])` to two decimals.
5. In one sentence, why scale by `1/sqrt(d_k)`?
