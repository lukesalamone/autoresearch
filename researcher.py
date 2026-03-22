#!/usr/bin/env python3
"""
researcher.py — LLM-based experiment planner with tool use.

Runs an agentic loop against a local Ollama model (qwen3-coder:30b):
  - read_file        — inspect train.py or results.txt before modifying
  - write_file       — rewrite train.py with a new training loop
  - propose_experiments — finalize the experiment queue (structured, no text parsing)

Called by the scheduler once per cycle with a 5-minute budget.
GPU is freed after the Ollama call via keep_alive=0.
"""

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import requests

# ── Config ─────────────────────────────────────────────────────────────────────

RESULTS_FILE   = "results.txt"
TODO_LOG       = "todo.log"
PROMPT_FILE    = "researcher_prompt.txt"
RESEARCHER_LOG = "logs/researcher.log"

OLLAMA_URL     = "http://localhost:11434/api/chat"
MODEL          = "qwen3-coder:30b"
OLLAMA_TIMEOUT = 240  # seconds per individual Ollama call; loop may make several

# Files the model is allowed to overwrite (relative to cwd)
WRITEABLE_PATHS = {"train.py"}

# Max iterations in the tool-use loop before giving up
MAX_ITERATIONS = 10


# ── Tools ──────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. "
                "Use this to inspect train.py before deciding whether to modify it, "
                "or to re-read results.txt for details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path, e.g. 'train.py' or 'results.txt'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Overwrite a file with new content. "
                "Use this to rewrite train.py when you want to add a new distillation objective, "
                "change the student architecture, or fix a bug in the training loop. "
                "The previous version is automatically backed up. "
                f"Allowed paths: {sorted(WRITEABLE_PATHS)}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to write. Must be one of the allowed paths.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete new file content.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of what changed and why.",
                    },
                },
                "required": ["path", "content", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_experiments",
            "description": (
                "Submit the experiment queue for this cycle. "
                "Call this exactly once when you are ready — it ends the planning phase. "
                "Experiments run sequentially; total cutoff_minutes must sum to ≤50."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": "3–6 sentence summary: best result so far, patterns observed, what you're trying next.",
                    },
                    "experiments": {
                        "type": "array",
                        "description": "Ordered list of experiments to run.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cutoff_minutes":    {"type": "number"},
                                "layers":            {"type": "integer"},
                                "hidden_dim":        {"type": "integer"},
                                "num_heads":         {"type": "integer"},
                                "intermediate_dim":  {"type": "integer"},
                                "temperature":       {"type": "number"},
                                "alpha":             {"type": "number"},
                                "distill_objective": {
                                    "type": "string",
                                    "enum": ["logit_kl", "attention_transfer", "hidden_transfer", "combined"],
                                },
                                "lr":               {"type": "number"},
                                "weight_decay":     {"type": "number"},
                                "batch_size":       {"type": "integer"},
                                "max_seq_len":      {"type": "integer"},
                                "dataset_fraction": {"type": "number"},
                                "notes":            {"type": "string"},
                            },
                            "required": [
                                "cutoff_minutes", "layers", "hidden_dim", "num_heads",
                                "distill_objective", "lr", "notes",
                            ],
                        },
                    },
                },
                "required": ["analysis", "experiments"],
            },
        },
    },
]


# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] RESEARCHER: {msg}"
    print(line, flush=True)
    try:
        os.makedirs("logs", exist_ok=True)
        with open(RESEARCHER_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── Tool implementations ───────────────────────────────────────────────────────

def tool_read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: {path} does not exist"
    try:
        content = p.read_text()
        log(f"read_file: {path} ({len(content)} chars)")
        return content
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def tool_write_file(path: str, content: str, reason: str) -> str:
    if path not in WRITEABLE_PATHS:
        msg = f"ERROR: '{path}' is not in the allowed write list {sorted(WRITEABLE_PATHS)}"
        log(msg)
        return msg

    p = Path(path)

    # Back up existing file
    if p.exists():
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup  = Path("logs") / f"{p.stem}_backup_{ts}{p.suffix}"
        os.makedirs("logs", exist_ok=True)
        shutil.copy2(p, backup)
        log(f"write_file: backed up {path} → {backup}")

    p.write_text(content)
    log(f"write_file: wrote {path} ({len(content)} chars) — reason: {reason}")
    return f"OK: wrote {len(content)} chars to {path}"


def dispatch_tool(name: str, args: dict) -> str:
    if name == "read_file":
        return tool_read_file(args["path"])
    elif name == "write_file":
        return tool_write_file(args["path"], args["content"], args.get("reason", ""))
    elif name == "propose_experiments":
        # Handled by the caller; return acknowledgement
        return "OK: experiments recorded"
    else:
        return f"ERROR: unknown tool '{name}'"


# ── Ollama agentic loop ────────────────────────────────────────────────────────

def call_ollama(messages: list, last_call: bool = False) -> dict:
    """Single Ollama chat call. Returns the raw response dict."""
    payload = {
        "model":      MODEL,
        "stream":     False,
        "keep_alive": 0 if last_call else -1,  # free VRAM only on the final call
        "messages":   messages,
        "tools":      TOOLS,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        log(f"Ollama call timed out after {OLLAMA_TIMEOUT}s")
        raise
    except requests.exceptions.ConnectionError:
        log(f"Could not connect to Ollama at {OLLAMA_URL} — is it running?")
        raise


def run_agent(system_prompt: str, user_message: str):
    """
    Agentic loop: keep calling Ollama and executing tool calls until
    propose_experiments is invoked or MAX_ITERATIONS is reached.
    Returns the experiments list, or None on failure.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    proposed_experiments = None
    proposed_analysis    = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"Agent iteration {iteration}/{MAX_ITERATIONS}")

        is_last = (iteration == MAX_ITERATIONS)
        data    = call_ollama(messages, last_call=is_last)

        message     = data["message"]
        tool_calls  = message.get("tool_calls") or []
        text_content = message.get("content", "").strip()

        # Strip Qwen3 <think>...</think> blocks from logged text
        clean_text = re.sub(r"<think>.*?</think>", "", text_content, flags=re.DOTALL).strip()
        if clean_text:
            log(f"Model text: {clean_text[:200]}{'...' if len(clean_text) > 200 else ''}")

        # Append assistant turn to history
        messages.append({"role": "assistant", "content": text_content, "tool_calls": tool_calls})

        if not tool_calls:
            if proposed_experiments is not None:
                break  # already done
            if clean_text:
                # Model wrote analysis but forgot to call propose_experiments — nudge it
                log("Model produced text but no tool call — nudging to call propose_experiments")
                messages.append({
                    "role": "user",
                    "content": "Good analysis. Now please call propose_experiments with your experiment plan.",
                })
                continue
            log("No tool calls and no text — agent stalled")
            break

        # Execute each tool call
        for tc in tool_calls:
            fn   = tc["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            log(f"Tool call: {name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items() if k != 'content')})")

            if name == "propose_experiments":
                proposed_experiments = args.get("experiments", [])
                proposed_analysis    = args.get("analysis", "")
                # Don't break — let it finish the tool_calls list, then we stop
                result = "OK: experiments recorded, planning phase complete"
            else:
                result = dispatch_tool(name, args)

            messages.append({"role": "tool", "content": result})

        if proposed_experiments is not None:
            # Make sure the final Ollama call uses keep_alive=0 to free VRAM
            if not is_last:
                log("propose_experiments received — freeing Ollama VRAM")
                try:
                    requests.post(
                        OLLAMA_URL,
                        json={"model": MODEL, "keep_alive": 0, "messages": []},
                        timeout=10,
                    )
                except Exception:
                    pass
            break

    return proposed_experiments, proposed_analysis


# ── Experiment post-processing ─────────────────────────────────────────────────

def next_exp_number(results_text: str) -> int:
    ids = [int(m) for m in re.findall(r"exp_(\d+)", results_text)]
    return max(ids, default=0) + 1


def validate_and_assign_ids(experiments: list, start_id: int) -> list:
    valid   = []
    counter = start_id
    for exp in experiments:
        h  = exp.get("hidden_dim", 0)
        nh = exp.get("num_heads", 1)
        if h % nh != 0:
            log(f"Dropping experiment: hidden_dim={h} not divisible by num_heads={nh}")
            continue
        if exp.get("cutoff_minutes", 0) < 1:
            log(f"Dropping experiment: cutoff_minutes too short")
            continue
        exp["exp_id"] = f"exp_{counter:03d}"
        counter += 1
        valid.append(exp)
    return valid


def write_todo(experiments: list):
    with open(TODO_LOG, "w") as f:
        f.write(f"# Generated by researcher.py at {datetime.now().isoformat()}\n")
        for exp in experiments:
            f.write(json.dumps(exp) + "\n")
    log(f"Wrote {len(experiments)} experiments to {TODO_LOG}")


# ── Seed experiments (cycle 0, no results yet) ─────────────────────────────────

# Seed sizes (vocab=30,522, factorized embeddings, fp16):
#   embed=32, hidden=128, 2L: ~2.6MB  embed=64, hidden=256, 4L: ~8.2MB
SEED_EXPERIMENTS = [
    # Pretrained init baseline — fastest to converge, known-good architecture
    {"cutoff_minutes": 12, "embed_dim": 128, "hidden_dim": 128, "num_layers": 2,
     "num_heads": 2, "intermediate_dim": 512, "pooling": "mean",
     "lr": 5e-5, "warmup_ratio": 0.1, "batch_size": 64, "max_seq_len": 128,
     "infonce_temp": 0.05, "dataset_fraction": 1.0, "init_pretrained": True,
     "notes": "seed: pretrained init (L-2_H-128), mean pool, InfoNCE baseline"},

    # Small factorized — lots of headroom, fast iterations
    {"cutoff_minutes": 10, "embed_dim": 32, "hidden_dim": 128, "num_layers": 2,
     "num_heads": 2, "intermediate_dim": 256, "pooling": "mean",
     "lr": 5e-5, "warmup_ratio": 0.1, "batch_size": 64, "max_seq_len": 128,
     "infonce_temp": 0.05, "dataset_fraction": 1.0, "init_pretrained": False,
     "notes": "seed: factorized embed=32, hidden=128, 2L from scratch — ~2.6MB"},

    # Medium factorized — more capacity, still well under 10MB
    {"cutoff_minutes": 12, "embed_dim": 48, "hidden_dim": 192, "num_layers": 3,
     "num_heads": 4, "intermediate_dim": 384, "pooling": "mean",
     "lr": 5e-5, "warmup_ratio": 0.1, "batch_size": 64, "max_seq_len": 128,
     "infonce_temp": 0.05, "dataset_fraction": 1.0, "init_pretrained": False,
     "notes": "seed: factorized embed=48, hidden=192, 3L — ~5.5MB"},

    # Larger factorized — near budget ceiling, compare quality vs smaller models
    {"cutoff_minutes": 12, "embed_dim": 64, "hidden_dim": 256, "num_layers": 4,
     "num_heads": 4, "intermediate_dim": 512, "pooling": "mean",
     "lr": 5e-5, "warmup_ratio": 0.1, "batch_size": 32, "max_seq_len": 128,
     "infonce_temp": 0.05, "dataset_fraction": 1.0, "init_pretrained": False,
     "notes": "seed: factorized embed=64, hidden=256, 4L — ~8.2MB"},
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log("Researcher started")

    results_text = Path(RESULTS_FILE).read_text().strip() if Path(RESULTS_FILE).exists() else ""

    # First cycle: skip LLM, write seeds directly
    if not results_text:
        log("No results yet — writing seed experiments (no LLM call needed)")
        experiments = [dict(s, exp_id=f"exp_{i:03d}") for i, s in enumerate(SEED_EXPERIMENTS, 1)]
        write_todo(experiments)
        return

    # Load prompt
    if not Path(PROMPT_FILE).exists():
        log(f"ERROR: {PROMPT_FILE} not found")
        sys.exit(1)
    system_prompt = Path(PROMPT_FILE).read_text().strip()

    # Build user message
    result_summary = (
        results_text if len(results_text) < 12_000
        else "...(oldest results truncated)...\n" + results_text[-12_000:]
    )
    start_id     = next_exp_number(results_text)
    user_message = (
        f"Here are all experiment results so far:\n\n"
        f"```\n{result_summary}\n```\n\n"
        f"Start any new exp_ids from {start_id}. "
        f"Use read_file to inspect train.py if you want to modify it, "
        f"then call propose_experiments when ready."
    )

    log(f"Calling Ollama agent (model={MODEL}, max_iterations={MAX_ITERATIONS})")

    # Save full conversation log
    os.makedirs("logs", exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/researcher_{ts}.txt"

    try:
        experiments, analysis = run_agent(system_prompt, user_message)
    except Exception as e:
        log(f"Agent failed: {e}")
        sys.exit(1)
    finally:
        # Always free VRAM regardless of success or failure
        try:
            requests.post(OLLAMA_URL, json={"model": MODEL, "keep_alive": 0, "messages": []}, timeout=10)
        except Exception:
            pass

    if not experiments:
        log("ERROR: No experiments returned by agent")
        sys.exit(1)

    valid = validate_and_assign_ids(experiments, start_id)
    if not valid:
        log("ERROR: All proposed experiments failed validation")
        sys.exit(1)

    if analysis:
        print("\n--- Researcher analysis ---", flush=True)
        print(analysis, flush=True)
        print("---\n", flush=True)

    write_todo(valid)
    log(f"Done. {len(valid)} valid experiments queued.")

    # Log the final experiment list
    with open(log_path, "w") as f:
        f.write(f"Analysis:\n{analysis}\n\nExperiments:\n")
        for exp in valid:
            f.write(json.dumps(exp) + "\n")


if __name__ == "__main__":
    main()
