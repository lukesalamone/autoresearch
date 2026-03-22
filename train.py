#!/usr/bin/env python3
"""
train.py — small bi-encoder for browser reranking.

Student: factorized-embedding transformer → mean-pooled dense embedding.
Loss:    InfoNCE (MultipleNegativesRankingLoss) with in-batch negatives.
Data:    ms_marco v1.1 — (query, positive_passage) pairs extracted from training split.
Eval:    MRR@10 on MSMARCO dev queries (ms_marco v1.1 validation split).

Researcher hook: modify the model, loss, or data loading below.
All hyperparams come in via CLI; results are appended to results.txt.
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.modeling_bert import BertEncoder
from datasets import load_dataset

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_FILE    = "results.txt"
TOKENIZER_NAME  = "bert-base-uncased"   # matches all-mpnet-base-v2 / all-MiniLM vocab
TRAIN_DATASET   = "ms_marco"
TRAIN_VERSION   = "v1.1"
EVAL_DATASET    = "ms_marco"
EVAL_VERSION    = "v1.1"


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # Experiment identity
    p.add_argument("--exp_id",           required=True)
    p.add_argument("--cutoff_minutes",   type=float, default=10.0)
    p.add_argument("--notes",            default="")

    # Architecture
    p.add_argument("--embed_dim",        type=int,   default=32,
                   help="Factorized embedding dim. vocab→embed_dim→hidden_dim. "
                        "Smaller = cheaper embeddings table.")
    p.add_argument("--hidden_dim",       type=int,   default=128,
                   help="Transformer hidden size.")
    p.add_argument("--num_layers",       type=int,   default=2)
    p.add_argument("--num_heads",        type=int,   default=2,
                   help="Attention heads. hidden_dim must be divisible by num_heads.")
    p.add_argument("--intermediate_dim", type=int,   default=256,
                   help="FFN intermediate size.")
    p.add_argument("--pooling",          default="mean",
                   choices=["mean", "cls", "weighted_mean"])
    p.add_argument("--init_pretrained",  action="store_true",
                   help="Init encoder from google/bert_uncased_L-2_H-128_A-2. "
                        "Forces hidden_dim=128, num_layers=2, num_heads=2.")

    # Training
    p.add_argument("--lr",               type=float, default=5e-5)
    p.add_argument("--warmup_ratio",     type=float, default=0.1)
    p.add_argument("--weight_decay",     type=float, default=0.01)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--max_seq_len",      type=int,   default=128)
    p.add_argument("--infonce_temp",     type=float, default=0.05,
                   help="InfoNCE temperature. Lower = sharper distribution over in-batch negatives.")
    p.add_argument("--dataset_fraction", type=float, default=1.0,
                   help="Fraction of ms_marco v1.1 training split to use (full = ~82K examples, ~1280 batches/epoch at bs=64).")
    p.add_argument("--l2_reg_weight",    type=float, default=0.0,
                   help="L2 regularization weight on embeddings.")

    # Eval
    p.add_argument("--eval_queries",     type=int,   default=500,
                   help="Number of MSMARCO dev queries for MRR@10 eval.")
    p.add_argument("--eval_batch_size",  type=int,   default=256)

    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# ── Model ──────────────────────────────────────────────────────────────────────

class FactorizedEmbeddings(nn.Module):
    """
    Token embeddings at embed_dim (cheap), projected up to hidden_dim.
    Position and type embeddings are directly at hidden_dim.

    Size: vocab×embed_dim + embed_dim×hidden_dim + pos×hidden_dim
    vs standard: vocab×hidden_dim + pos×hidden_dim
    Saving: vocab×(hidden_dim - embed_dim) - embed_dim×hidden_dim params
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 max_position_embeddings=512, dropout=0.1):
        super().__init__()
        self.word_embeddings     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        # no type embeddings — not needed for single-sequence encoding
        self.projection = (
            nn.Linear(embed_dim, hidden_dim, bias=False)
            if embed_dim != hidden_dim else nn.Identity()
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # position ids buffer
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        word_emb = self.projection(self.word_embeddings(input_ids))
        pos_emb  = self.position_embeddings(self.position_ids[:, :seq_len])
        return self.dropout(self.norm(word_emb + pos_emb))


class SmallBiEncoder(nn.Module):
    """
    Factorized-embedding transformer bi-encoder.
    encode() returns L2-normalised embeddings; similarity = dot product.
    """
    def __init__(self, args, vocab_size):
        super().__init__()

        if args.hidden_dim % args.num_heads != 0:
            raise ValueError(f"hidden_dim {args.hidden_dim} must be divisible by num_heads {args.num_heads}")

        self.pooling = args.pooling

        self.embeddings = FactorizedEmbeddings(
            vocab_size   = vocab_size,
            embed_dim    = args.embed_dim,
            hidden_dim   = args.hidden_dim,
        )

        encoder_config = BertConfig(
            hidden_size             = args.hidden_dim,
            num_hidden_layers       = args.num_layers,
            num_attention_heads     = args.num_heads,
            intermediate_size       = args.intermediate_dim,
            hidden_dropout_prob     = 0.1,
            attention_probs_dropout_prob = 0.1,
            attn_implementation     = "eager",
        )
        self.encoder = BertEncoder(encoder_config)

        # Weighted mean pooling: one learned scalar per layer
        if args.pooling == "weighted_mean":
            self.layer_weights = nn.Parameter(torch.ones(args.num_layers))

    def _extend_mask(self, attention_mask):
        """0/1 mask → BertEncoder extended mask (0 or -10000)."""
        return (1.0 - attention_mask[:, None, None, :].float()) * -10000.0

    def encode(self, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        ext_mask = self._extend_mask(attention_mask)

        out = self.encoder(
            hidden_states      = hidden,
            attention_mask     = ext_mask,
            output_hidden_states = (self.pooling == "weighted_mean"),
        )

        if self.pooling == "cls":
            pooled = out.last_hidden_state[:, 0]

        elif self.pooling == "mean":
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        elif self.pooling == "weighted_mean":
            weights = F.softmax(self.layer_weights, dim=0)
            # out.hidden_states is a tuple: (embedding_output, layer1, ..., layerN)
            # skip index 0 (embedding output before first layer)
            stacked = torch.stack(out.hidden_states[1:], dim=0)  # (L, B, S, H)
            hidden_weighted = (stacked * weights[:, None, None, None]).sum(0)
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_weighted * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        return F.normalize(pooled, dim=-1)

    def forward(self, input_ids, attention_mask):
        return self.encode(input_ids, attention_mask)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

def model_size_mb(model):
    return count_params(model) * 2 / (1024 ** 2)  # fp16


# ── Loss ───────────────────────────────────────────────────────────────────────

def infonce_loss(q_emb, p_emb, temperature):
    """
    InfoNCE / MultipleNegativesRankingLoss.
    q_emb: (B, H), p_emb: (B, H) — both L2-normalised.
    Each query's positive is the diagonal; all other passages are in-batch negatives.
    Scaled dot-product cross-entropy: lower temperature = sharper, harder negatives.
    """
    logits = q_emb @ p_emb.T / temperature   # (B, B)
    labels = torch.arange(len(q_emb), device=q_emb.device)
    return F.cross_entropy(logits, labels)


def hybrid_loss(q_emb, p_emb, temperature, l2_reg_weight=0.0):
    """
    Hybrid loss: InfoNCE + L2 regularization on embeddings.
    """
    infonce = infonce_loss(q_emb, p_emb, temperature)
    
    # Add L2 regularization on embeddings
    l2_reg = 0.0
    if l2_reg_weight > 0:
        # Regularize the factorized embeddings (word embeddings + projection)
        # We regularize the word embeddings directly since they're the main parameters
        word_emb = q_emb.new_zeros(1)  # placeholder to avoid errors
        # Actually, we don't want to regularize the output embeddings, only the input embeddings
        # But we need to be careful about which parameters to regularize
        # Let's just regularize the word embeddings directly for now
        pass
    
    # For now, let's regularize the embeddings in a simpler way
    # Regularize the word embeddings directly
    return infonce


# ── Dataset ────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """Generic (query, positive_passage) pair dataset. Tokenizes on the fly."""
    def __init__(self, pairs, tokenizer, max_seq_len):
        self.pairs       = pairs
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, passage = self.pairs[idx]
        q_enc = self.tokenizer(query,   truncation=True, max_length=self.max_seq_len,
                               padding="max_length", return_tensors="pt")
        p_enc = self.tokenizer(passage, truncation=True, max_length=self.max_seq_len,
                               padding="max_length", return_tensors="pt")
        return {
            "q_input_ids": q_enc["input_ids"].squeeze(0),
            "q_mask":      q_enc["attention_mask"].squeeze(0),
            "p_input_ids": p_enc["input_ids"].squeeze(0),
            "p_mask":      p_enc["attention_mask"].squeeze(0),
        }


def extract_msmarco_pairs(hf_dataset):
    """(query, first selected passage) from ms_marco v1.1."""
    pairs = []
    for row in hf_dataset:
        query    = row["query"]
        passages = row["passages"]["passage_text"]
        selected = row["passages"]["is_selected"]
        for text, is_sel in zip(passages, selected):
            if is_sel:
                pairs.append((query, text))
                break
    return pairs


def extract_squad_pairs(hf_dataset):
    """(question, context paragraph) from SQuAD. Each context is a relevant passage."""
    pairs = []
    seen  = set()
    for row in hf_dataset:
        key = (row["question"], row["context"])
        if key not in seen:
            pairs.append(key)
            seen.add(key)
    return pairs


def load_train_data(tokenizer, max_seq_len, dataset_fraction, batch_size, seed):
    # MS MARCO
    print(f"Loading {TRAIN_DATASET} {TRAIN_VERSION} train split...", flush=True)
    msmarco_ds = load_dataset(TRAIN_DATASET, TRAIN_VERSION, split="train")
    n = max(1, int(len(msmarco_ds) * dataset_fraction))
    msmarco_ds = msmarco_ds.select(range(n))
    msmarco_pairs = extract_msmarco_pairs(msmarco_ds)
    print(f"  MS MARCO: {len(msmarco_pairs):,} pairs from {n:,} examples", flush=True)

    # SQuAD
    print("Loading SQuAD train split...", flush=True)
    squad_ds    = load_dataset("squad", split="train")
    n_squad     = max(1, int(len(squad_ds) * dataset_fraction))
    squad_ds    = squad_ds.select(range(n_squad))
    squad_pairs = extract_squad_pairs(squad_ds)
    print(f"  SQuAD:    {len(squad_pairs):,} pairs from {n_squad:,} examples", flush=True)

    all_pairs = msmarco_pairs + squad_pairs
    print(f"  Total:    {len(all_pairs):,} pairs", flush=True)

    dataset    = PairDataset(all_pairs, tokenizer, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    return dataloader


# ── Eval: MRR@10 on MSMARCO dev ───────────────────────────────────────────────

def load_eval_data(n_queries, seed):
    """
    Load n_queries examples from the MSMARCO dev split.
    Each example has a query + ~10 candidate passages (1 relevant).
    Returns list of (query_text, [passage_text], [is_relevant]).
    """
    print(f"Loading MSMARCO dev eval set ({n_queries} queries)...", flush=True)
    ds = load_dataset(EVAL_DATASET, EVAL_VERSION, split="validation")

    # Deterministic shuffle then take n_queries
    ds = ds.shuffle(seed=seed).select(range(min(n_queries, len(ds))))

    eval_data = []
    for row in ds:
        query    = row["query"]
        passages = row["passages"]["passage_text"]
        labels   = row["passages"]["is_selected"]
        if sum(labels) == 0:
            continue  # skip queries with no relevant passage
        eval_data.append((query, passages, labels))

    print(f"  Loaded {len(eval_data)} eval queries", flush=True)
    return eval_data


@torch.no_grad()
def compute_mrr10(model, eval_data, tokenizer, max_seq_len, batch_size, device):
    """MRR@10 over MSMARCO dev candidates."""
    model.eval()

    def encode_texts(texts):
        enc = tokenizer(
            texts, truncation=True, max_length=max_seq_len,
            padding=True, return_tensors="pt",
        )
        embeddings = []
        for i in range(0, len(texts), batch_size):
            ids  = enc["input_ids"][i:i+batch_size].to(device)
            mask = enc["attention_mask"][i:i+batch_size].to(device)
            embeddings.append(model.encode(ids, mask).cpu())
        return torch.cat(embeddings, dim=0)

    reciprocal_ranks = []
    for query, passages, labels in eval_data:
        q_emb = encode_texts([query])              # (1, H)
        p_emb = encode_texts(passages)             # (P, H)
        scores = (q_emb @ p_emb.T).squeeze(0)     # (P,)
        ranked = scores.argsort(descending=True).tolist()

        rr = 0.0
        for rank, idx in enumerate(ranked[:10], start=1):
            if labels[idx]:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args, model, train_loader, eval_data, tokenizer, device):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps   = len(train_loader) * 1000  # effectively infinite; cutoff handles it
    warmup_steps  = int(total_steps * args.warmup_ratio)
    lr_scheduler  = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    cutoff_sec = args.cutoff_minutes * 60
    start      = time.time()
    step       = 0
    epoch      = 0
    mrr_scores = []
    trajectory = "too_short"

    model.train()

    while True:
        epoch += 1
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            if time.time() - start >= cutoff_sec:
                break

            q_emb = model.encode(batch["q_input_ids"].to(device), batch["q_mask"].to(device))
            p_emb = model.encode(batch["p_input_ids"].to(device), batch["p_mask"].to(device))
            
            # Use hybrid loss
            if args.l2_reg_weight > 0:
                # Simple hybrid: InfoNCE + L2 on word embeddings
                loss = infonce_loss(q_emb, p_emb, args.infonce_temp)
                
                # Add L2 regularization on the embeddings
                l2_reg = 0.0
                for name, param in model.named_parameters():
                    if 'word_embeddings' in name:
                        l2_reg += param.norm(2).pow(2)
                
                loss = loss + args.l2_reg_weight * l2_reg / 2.0
            else:
                loss = infonce_loss(q_emb, p_emb, args.infonce_temp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            step        += 1
            epoch_loss  += loss.item()
            epoch_steps += 1

            if step % 200 == 0:
                elapsed = time.time() - start
                print(
                    f"  step={step:5d}  loss={epoch_loss/epoch_steps:.4f}  "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

        if time.time() - start >= cutoff_sec:
            break

        # Val eval at end of each epoch
        mrr = compute_mrr10(model, eval_data, tokenizer, args.max_seq_len,
                            args.eval_batch_size, device)
        mrr_scores.append(mrr)
        print(f"  [epoch {epoch}] mrr@10={mrr:.4f}", flush=True)
        model.train()

    # Trajectory
    if len(mrr_scores) >= 3:
        r = mrr_scores[-3:]
        if r[-1] > r[0] * 1.01:
            trajectory = "improving"
        elif r[-1] < r[0] * 0.99:
            trajectory = "diverging"
        else:
            trajectory = "plateaued"
    elif len(mrr_scores) == 1:
        trajectory = "single_eval"

    final_mrr = mrr_scores[-1] if mrr_scores else 0.0
    wall_time = time.time() - start
    return final_mrr, trajectory, wall_time, step


# ── Result logging ─────────────────────────────────────────────────────────────

def write_result(args, mrr, trajectory, wall_time, size_mb, steps):
    hparams = {
        "embed_dim":        args.embed_dim,
        "hidden_dim":       args.hidden_dim,
        "num_layers":       args.num_layers,
        "num_heads":        args.num_heads,
        "intermediate_dim": args.intermediate_dim,
        "pooling":          args.pooling,
        "lr":               args.lr,
        "warmup_ratio":     args.warmup_ratio,
        "batch_size":       args.batch_size,
        "max_seq_len":      args.max_seq_len,
        "infonce_temp":     args.infonce_temp,
        "dataset_fraction": args.dataset_fraction,
        "cutoff_minutes":   args.cutoff_minutes,
        "init_pretrained":  args.init_pretrained,
        "l2_reg_weight":    args.l2_reg_weight,
    }
    sep = "=" * 60
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n{sep}\n")
        f.write(f"exp_id:             {args.exp_id}\n")
        f.write(f"timestamp:          {datetime.now().isoformat()}\n")
        f.write(f"mrr@10:             {mrr:.4f}\n")
        f.write(f"trajectory:         {trajectory}\n")
        f.write(f"wall_time_s:        {wall_time:.0f}\n")
        f.write(f"steps:              {steps}\n")
        f.write(f"model_size_mb_fp16: {size_mb:.2f}\n")
        f.write(f"hyperparams:        {json.dumps(hparams)}\n")
        if args.notes:
            f.write(f"notes:              {args.notes}\n")
    print(f"Result written to {RESULTS_FILE}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"[{args.exp_id}] device={device}  cutoff={args.cutoff_minutes}min", flush=True)
    if args.notes:
        print(f"  notes: {args.notes}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size

    # If init_pretrained, force matching architecture
    if args.init_pretrained:
        if args.hidden_dim != 128 or args.num_layers != 2 or args.num_heads != 2:
            print(
                "  WARNING: init_pretrained forces hidden_dim=128, num_layers=2, num_heads=2",
                flush=True,
            )
        args.hidden_dim  = 128
        args.num_layers  = 2
        args.num_heads   = 2
        args.embed_dim   = 128  # no factorization when loading pretrained weights

    # Build student
    model   = SmallBiEncoder(args, vocab_size).to(device)
    size_mb = model_size_mb(model)
    print(f"  Student: {count_params(model)/1e6:.3f}M params  {size_mb:.2f}MB fp16", flush=True)
    if size_mb > 10.0:
        print(f"  WARNING: {size_mb:.2f}MB exceeds 10MB target", flush=True)

    # Optionally init encoder weights from pretrained tiny BERT
    if args.init_pretrained:
        print("  Loading pretrained weights from google/bert_uncased_L-2_H-128_A-2...", flush=True)
        from transformers import BertModel
        pretrained = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        # Load encoder layers (embeddings are incompatible due to factorization; skip them)
        missing, unexpected = model.encoder.load_state_dict(
            pretrained.encoder.state_dict(), strict=False
        )
        print(f"  Loaded encoder. Missing: {len(missing)}  Unexpected: {len(unexpected)}", flush=True)
        del pretrained

    # Data
    train_loader = load_train_data(
        tokenizer, args.max_seq_len, args.dataset_fraction,
        args.batch_size, args.seed,
    )
    eval_data = load_eval_data(args.eval_queries, args.seed)

    # Train
    mrr, trajectory, wall_time, steps = train(
        args, model, train_loader, eval_data, tokenizer, device
    )

    print(
        f"[{args.exp_id}] DONE  mrr@10={mrr:.4f}  "
        f"trajectory={trajectory}  time={wall_time:.0f}s  steps={steps}",
        flush=True,
    )
    write_result(args, mrr, trajectory, wall_time, size_mb, steps)


if __name__ == "__main__":
    main()