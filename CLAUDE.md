# Retrieval Model Distillation Autoresearch

## Goal

Distill a sentence-transformer retrieval model (e.g. `all-mpnet-base-v2`, `all-MiniLM-L12-v2`) into a ~10MB student model using automated, overnight experiment loops. Based on Karpathy's autoresearch pattern: an LLM agent iteratively edits `train.py`, runs short training experiments, and keeps changes that improve the validation ranking metric (NDCG@10).

The student is a bi-encoder: a small transformer encoder + mean pooling that produces dense embeddings for retrieval. It must fit in ~10MB (fp16) while retaining as much retrieval quality as possible.

## Problem Setup

**Teacher**: a pre-trained sentence-transformer retrieval model. Good choices:
- `sentence-transformers/all-mpnet-base-v2` — stronger, 420MB
- `sentence-transformers/all-MiniLM-L12-v2` — smaller teacher, 120MB, already quite compressed

**Student**: a custom small `BertModel` + mean pooling, built from scratch. Target size ≤10MB fp16.

**Training data**: MS MARCO passage ranking with pre-computed cross-encoder scores (e.g. `msmarco-scores-ms-marco-MiniLM-L6-v2`). Each example is a (query, passage, score) triple where score is a cross-encoder soft label — much richer signal than binary relevance.

**Distillation objectives** (things the researcher can explore):
- MarginMSE: MSE on score differences between passage pairs per query
- KL divergence on softmax of scores per query (treats scores as a distribution)
- Embedding MSE: directly match student embeddings to teacher embeddings
- Contrastive with soft labels: InfoNCE weighted by teacher scores

**Validation metric**: NDCG@10 on a BEIR out-of-domain task (e.g. TREC-COVID or NFCorpus). Out-of-domain eval is intentional — it measures generalisation, not just in-distribution fitting.

## Architecture

**Batched planning loop** — alternates between GPU-intensive training and LLM-based planning:

1. **Train phase**: Run a batch of N experiments sequentially on GPU (each ~5–15 min). No LLM running during this phase.
2. **Plan phase**: Local Ollama LLM (`qwen3-coder:30b`) reads the full experiment log, optionally rewrites `train.py`, and proposes the next batch of experiments via structured tool calls.
3. Repeat until morning or convergence.

This avoids GPU contention between training and the local LLM.

## Key Design Decisions

- **Local LLM planner**: `qwen3-coder:30b` via Ollama. No API cost, no data leaving the machine.
- **Tool use for code editing**: the researcher can call `read_file` / `write_file` to modify `train.py` directly, not just tune hyperparameters. This lets it add new loss functions, change pooling strategies, etc.
- **Batch analysis over per-experiment feedback**: seeing N results at once lets the planner spot hyperparameter interactions that sequential adaptation would miss.
- **Soft labels over binary relevance**: cross-encoder scores give dense training signal; binary MS MARCO labels (1 relevant per query) would starve the distillation loss.

## GPU Time-Sharing

The researcher (Ollama) and the training jobs both run on the same GPU. They must never overlap. The scheduler enforces strict alternation:

1. **Researcher phase** (5 min): Ollama loads `qwen3-coder:30b`, reads results, proposes experiments. `keep_alive=0` is set on the request so Ollama unloads the model from VRAM immediately when the response is returned — not after a timeout.
2. **Training phase** (55 min): `train.py` subprocesses run sequentially. Each subprocess owns the full GPU. When a subprocess exits, the OS releases its CUDA context before the next one starts.
3. No overlap is possible by construction: the scheduler is single-threaded and calls each phase with `subprocess.run()` (blocking).

If Ollama fails to unload (e.g. crash), training will OOM. The scheduler does not currently detect this — if experiments start failing with CUDA OOM, check `nvidia-smi` and kill stale Ollama processes manually.

## Experiment Log Format

Each result block in `results.txt` includes:

- Experiment ID
- All hyperparameters (lr, layers, hidden_dim, loss type, pooling, etc.)
- `ndcg@10` on the validation BEIR task (primary metric, higher is better)
- Trajectory: improving / plateaued / diverging at cutoff
- Wall time and gradient steps

## Size Constraint

Teacher vocab determines the embedding table size. For `all-mpnet-base-v2` / `all-MiniLM-L12-v2` the tokenizer is `bert-base-uncased` (vocab = 30,522).

Example student sizes (fp16):
- hidden=128: embeddings=7.4MB, 2 layers≈0.5MB → ~7.9MB ✓
- hidden=256: embeddings=14.8MB alone → over budget ✗

The 10MB limit is a hard constraint. Architectural changes (smaller hidden, fewer layers, tied embeddings, quantisation) matter more than hyperparameter tuning.

## Risks and Constraints

- **Short runs can mislead.** 5-minute experiments may not complete enough gradient steps to distinguish good configs from bad ones. Trajectory shape (still improving vs. plateaued) is as important as the final metric.
- **BEIR eval is slow.** Running full NDCG@10 eval takes time; consider a small proxy eval set for short experiments and full eval only for promising configs.
- **The planner is not omniscient.** It will beat random search but won't converge cleanly. Expect useful direction, not optimal solutions.

## Cost Budget

- GPU: fully utilised overnight
- LLM planning calls: local Ollama, no API cost
- Target: meaningful improvement in NDCG@10 by morning with zero babysitting
