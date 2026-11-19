# AI & LLM Deep Dive

A 52-week, hands-on path from linear algebra to training, aligning and shipping large language models, at 5–8 hours a week.
Each week has a lesson, a practice notebook you can run in Google Colab with one click, and a build task.

- **Learning Hub** (progress, highlights, notes, tutor): https://claude.ai/artifact/VwTfCas4FMnB23QHuBUJhj
- **Started:** 16 Nov 2026 · **Finishes:** 14 Nov 2027
- **Author:** Karim Aly, PhD (validation engineering · deep-learning speech recognition)

## How the notebooks work

1. Click a **Open in Colab** badge. Colab opens a copy of the notebook from this repo.
2. Use **File → Save a copy in Drive** so your work is kept.
3. Each exercise has a starter cell with `TODO`s and a **Check** cell. Solutions are at the bottom of each notebook.
4. Put your own finished versions in [`my-work/`](my-work/) and commit them; that is the portfolio.

Run locally instead: `pip install -r requirements.txt`, then `jupyter lab`.

## Weekly plan

### Phase 0 · Math & ML refresh (weeks 1–4)
_Gate: Training loop written from memory_

| Week | Starts | Topic | Lesson | Notebook |
| --- | --- | --- | --- | --- |
| 1 | 16 Nov 2026 | Linear algebra for attention | [read](lessons/week-01.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuranLearning/ai-llm-deep-dive/blob/main/notebooks/week-01-linear-algebra.ipynb) |
| 2 | 23 Nov 2026 | Calculus, probability and backprop by hand | [read](lessons/week-02.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuranLearning/ai-llm-deep-dive/blob/main/notebooks/week-02-backprop-by-hand.ipynb) |
| 3 | 30 Nov 2026 | NumPy MLP and the Adam optimizer | [read](lessons/week-03.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuranLearning/ai-llm-deep-dive/blob/main/notebooks/week-03-numpy-mlp-adam.ipynb) |
| 4 · gate | 07 Dec 2026 | PyTorch and the training loop | [read](lessons/week-04.md) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuranLearning/ai-llm-deep-dive/blob/main/notebooks/week-04-pytorch-training-loop.ipynb) |

### Phase 1 · Transformers from scratch (weeks 5–14)
_Gate: Your own GPT trained on a small corpus_

| Week | Starts | Topic | Lesson | Notebook |
| --- | --- | --- | --- | --- |
| 5 | 14 Dec 2026 | micrograd: an autograd engine, part 1 | coming | coming |
| 6 | 21 Dec 2026 | micrograd, part 2 | coming | coming |
| 7 | 28 Dec 2026 | makemore: bigram and MLP language models | coming | coming |
| 8 | 04 Jan 2027 | Activations, BatchNorm and manual backprop | coming | coming |
| 9 | 11 Jan 2027 | WaveNet-style models | coming | coming |
| 10 | 18 Jan 2027 | Self-attention from scratch | coming | coming |
| 11 | 25 Jan 2027 | Building GPT | coming | coming |
| 12 | 01 Feb 2027 | Tokenization and BPE | coming | coming |
| 13 | 08 Feb 2027 | Reproducing GPT-2, part 1 | coming | coming |
| 14 · gate | 15 Feb 2027 | Reproducing GPT-2, part 2 | coming | coming |

### Phase 2 · Pretraining & scaling (weeks 15–22)
_Gate: Tokenizer + scaling-law study + nanochat run_

| Week | Starts | Topic | Lesson | Notebook |
| --- | --- | --- | --- | --- |
| 15 | 22 Feb 2027 | CS336: overview, tokenization, resources | coming | coming |
| 16 | 01 Mar 2027 | CS336: transformer architectures | coming | coming |
| 17 | 08 Mar 2027 | Mixture of experts and GPUs | coming | coming |
| 18 | 15 Mar 2027 | Kernels and FlashAttention | coming | coming |
| 19 | 22 Mar 2027 | Parallelism | coming | coming |
| 20 | 29 Mar 2027 | Scaling laws | coming | coming |
| 21 | 05 Apr 2027 | Pretraining data | coming | coming |
| 22 · gate | 12 Apr 2027 | End-to-end: nanochat | coming | coming |

### Phase 3 · Post-training (weeks 23–30)
_Gate: SFT + DPO + GRPO with before/after evals_

| Week | Starts | Topic | Lesson | Notebook |
| --- | --- | --- | --- | --- |
| 23 | 19 Apr 2027 | Reinforcement learning basics | coming | coming |
| 24 | 26 Apr 2027 | SFT and RLHF | coming | coming |
| 25 | 03 May 2027 | Parameter-efficient fine-tuning | coming | coming |
| 26 | 10 May 2027 | Evaluating chat models | coming | coming |
| 27 | 17 May 2027 | Direct Preference Optimization | coming | coming |
| 28 | 24 May 2027 | GRPO and RL for reasoning | coming | coming |
| 29 | 31 May 2027 | Reasoning models | coming | coming |
| 30 · gate | 07 Jun 2027 | Domain fine-tune: validation reports | coming | coming |

### Phase 4 · Applied LLM engineering (weeks 31–42)
_Gate: RAG + MCP agent with an eval suite in CI_

| Week | Starts | Topic | Lesson | Notebook |
| --- | --- | --- | --- | --- |
| 31 | 14 Jun 2027 | Prompting and structured outputs | coming | coming |
| 32 | 21 Jun 2027 | Tool use and prompt versioning | coming | coming |
| 33 | 28 Jun 2027 | RAG foundations | coming | coming |
| 34 | 05 Jul 2027 | Hybrid search and reranking | coming | coming |
| 35 | 12 Jul 2027 | Evaluating RAG | coming | coming |
| 36 | 19 Jul 2027 | Evaluation sets and LLM-as-judge | coming | coming |
| 37 | 26 Jul 2027 | Evals in CI | coming | coming |
| 38 | 02 Aug 2027 | Agent loops | coming | coming |
| 39 | 09 Aug 2027 | Model Context Protocol | coming | coming |
| 40 | 16 Aug 2027 | Multi-tool agents and agent evals | coming | coming |
| 41 | 23 Aug 2027 | Serving and quantization | coming | coming |
| 42 · gate | 30 Aug 2027 | Latency, cost and write-up | coming | coming |

### Phase 5 · Specialization & capstone (weeks 43–52)
_Gate: Capstone repo, eval table and public write-up_

| Week | Starts | Topic | Lesson | Notebook |
| --- | --- | --- | --- | --- |
| 43 | 06 Sep 2027 | Choose your capstone track | coming | coming |
| 44 | 13 Sep 2027 | Interpretability basics | coming | coming |
| 45 | 20 Sep 2027 | Multimodal models and capstone spec | coming | coming |
| 46 | 27 Sep 2027 | Capstone: data pipeline | coming | coming |
| 47 | 04 Oct 2027 | Capstone: baseline | coming | coming |
| 48 | 11 Oct 2027 | Capstone: iteration 1 | coming | coming |
| 49 | 18 Oct 2027 | Capstone: iteration 2 | coming | coming |
| 50 | 25 Oct 2027 | Capstone: demo and README | coming | coming |
| 51 | 01 Nov 2027 | Write it up | coming | coming |
| 52 · gate | 08 Nov 2027 | Retrospective and Year 2 | coming | coming |

## Sources

Lessons are original explanations. Lectures, books and papers are linked, never copied: Karpathy's *Neural Networks: Zero to Hero*, Stanford CS336, *Understanding Deep Learning*, *Speech and Language Processing*, the RLHF Book, Hugging Face courses and the original papers. See each lesson for the exact links.
