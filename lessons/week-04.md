# Week 4: PyTorch and the training loop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuranLearning/ai-llm-deep-dive/blob/main/notebooks/week-04-pytorch-training-loop.ipynb) · [Practice notebook](../notebooks/week-04-pytorch-training-loop.ipynb) · [Learning Hub week page](https://claude.ai/artifact/VwTfCas4FMnB23QHuBUJhj#w-ai-llm-4)

**This week:** Watch: [Learn PyTorch for Deep Learning · ch. 0–2](https://www.learnpytorch.io/) · Read: [The Little Book of Deep Learning · skim](https://fleuret.org/francois/lbdl.html) · Build: MLP in PyTorch; full training loop written from memory

---

Gate week. You move last week's MLP into PyTorch and finish by writing a complete training loop from memory. Every model you train for the next eleven months uses this loop, so it pays to make it automatic.

## 1. Tensors and autograd

A PyTorch tensor is a NumPy array that can live on a GPU and remember how it was computed. Set `requires_grad=True` on a parameter and every operation on it is recorded in a graph. Calling `.backward()` on a scalar loss runs the chain rule through that graph and fills `.grad` on each parameter: the same thing you did by hand in week 3.

```python
import torch
w = torch.tensor([2.0, -1.0], requires_grad=True)
x = torch.tensor([3.0, 4.0])
loss = (w @ x - 1) ** 2
loss.backward()
print(w.grad)          # 2·(w·x − 1)·x = 2·1·[3, 4] = [6, 8]
```

Two habits prevent most bugs:

- Gradients **accumulate**. Call `optimizer.zero_grad()` (or `set_to_none=True`) every step.
- Wrap evaluation in `torch.no_grad()` so no graph is built.

## 2. The same MLP as an nn.Module

```python
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, d, h, c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, c))
    def forward(self, x):
        return self.net(x)          # logits; the loss applies softmax
```

`nn.CrossEntropyLoss` takes raw logits and integer targets and fuses softmax with the negative log-likelihood in a numerically stable way. Never apply softmax yourself before it.

**Sanity check:** compare against your NumPy version. Copy the same initial weights into the module (`model.net[0].weight.data = torch.tensor(W1.T)`; note the transpose, since `nn.Linear` stores `(out, in)`) and confirm the loss and gradients match to about 1e-6.

## 3. The training loop, piece by piece

```python
import math, torch
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
val_dl   = DataLoader(TensorDataset(Xva, yva), batch_size=256)

model = MLP(d, 64, C).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

total = epochs * len(train_dl); warmup = int(0.05 * total)
def lr_at(step):                       # linear warmup, then cosine decay
    if step < warmup: return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

step, best = 0, float("inf")
for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1

    model.eval(); vl, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            vl += loss_fn(model(xb), yb).item() * len(yb); n += len(yb)
    vl /= n
    print(f"epoch {epoch}  val {vl:.4f}  lr {sched.get_last_lr()[0]:.2e}")
    if vl < best:
        best = vl
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "val": vl}, "ckpt.pt")
```

The pieces, and why each is there:

| Piece | Why |
| --- | --- |
| Seed | Runs you can repeat and compare |
| Shuffled train loader, fixed val loader | Unbiased batches; a stable yardstick |
| AdamW + weight decay | The LLM default optimizer (week 3) |
| Warmup + cosine schedule | Warmup avoids early instability; decay settles into a minimum. GPT training uses exactly this shape |
| `zero_grad` → `backward` → clip → `step` | The core four lines. Clipping caps rare exploding gradients |
| `model.train()` / `model.eval()` | Switch dropout and batch-norm behaviour |
| Checkpoint on best val | Resume after a crash; keep the best model |

## 4. Reading: The Little Book of Deep Learning

Skim the whole book this week (it is about 170 small pages). Its job is to give you a map: you will return to its chapters on attention and training when they come up.

## Gate 0 checklist

Mark this week Done when all of these are true:

- [ ] PyTorch MLP matches your NumPy gradients to about 1e-6.
- [ ] The MLP solves the spiral dataset (validation accuracy above 95%).
- [ ] You wrote the training loop above **from a blank file, without looking**, and it ran.
- [ ] You can explain each row of the table in your own words.
- [ ] Code pushed to a GitHub repo with a short README: problem, approach, result, what failed.

## Self-check

1. What goes wrong if you forget `zero_grad()`?
2. Why does `nn.CrossEntropyLoss` take logits rather than probabilities?
3. Sketch the learning-rate curve for warmup + cosine decay.
4. What two things should a checkpoint contain so training can resume exactly?
5. Why is `model.eval()` needed even for a model with no dropout, once you add batch-norm?
