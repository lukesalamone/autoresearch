#!/usr/bin/env python3
"""
scheduler.py — top-level overnight loop.

Each 60-minute cycle:
  1. Researcher phase (5 min):  call researcher.py to read results.txt,
                                propose next experiments to todo.log
  2. Experiment phase (55 min): read todo.log, run each experiment
                                sequentially until budget is exhausted

Run with:
    python3 scheduler.py 2>&1 | tee logs/scheduler_$(date +%Y%m%d_%H%M%S).log
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────

TODO_LOG          = "todo.log"
RESULTS_FILE      = "results.txt"
RESEARCHER_SCRIPT = "researcher.py"
TRAIN_SCRIPT      = "train.py"

RESEARCHER_MINUTES = 5
EXPERIMENT_MINUTES = 55


def parse_args():
    p = argparse.ArgumentParser(description="Auto-researcher scheduler")
    p.add_argument(
        "--experiment_time", type=float, default=None,
        metavar="MINUTES",
        help="Override each experiment's cutoff_minutes for this run. "
             "Useful for quick debugging (e.g. --experiment_time 2).",
    )
    return p.parse_args()

# Grace period added on top of each experiment's cutoff_minutes before hard kill
EXPERIMENT_GRACE_SECONDS = 120
EXPERIMENT_MIN_MINUTES   = 5     # hard floor; researcher must not go below this


# ── Logging ────────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] SCHEDULER: {msg}", flush=True)


# ── Researcher phase ───────────────────────────────────────────────────────────

def run_researcher(budget_sec):
    log(f"Researcher phase started ({budget_sec/60:.0f} min budget)")
    try:
        result = subprocess.run(
            [sys.executable, RESEARCHER_SCRIPT],
            timeout=budget_sec,
        )
        if result.returncode != 0:
            log(f"Researcher exited with non-zero code {result.returncode}")
        else:
            log("Researcher phase complete")
    except subprocess.TimeoutExpired:
        log("Researcher hit time budget (normal)")
    except FileNotFoundError:
        log(f"ERROR: {RESEARCHER_SCRIPT} not found")
    except Exception as e:
        log(f"Researcher error: {e}")


# ── Experiment phase ───────────────────────────────────────────────────────────

def read_todo():
    """Parse todo.log — one JSON object per line, lines starting with # ignored."""
    if not os.path.exists(TODO_LOG):
        return []
    experiments = []
    with open(TODO_LOG) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                experiments.append(json.loads(line))
            except json.JSONDecodeError as e:
                log(f"Skipping bad todo.log line {lineno}: {e}")
    return experiments


def clear_todo():
    with open(TODO_LOG, "w") as f:
        pass


def build_train_cmd(exp, exp_id, capped_cutoff_minutes):
    """Convert an experiment dict to a train.py command."""
    skip = {"exp_id", "cutoff_minutes"}
    cmd  = [
        sys.executable, TRAIN_SCRIPT,
        "--exp_id",           exp_id,
        "--cutoff_minutes",   str(capped_cutoff_minutes),
    ]
    for k, v in exp.items():
        if k in skip:
            continue
        if isinstance(v, bool):
            # store_true flags: include the flag if True, omit if False
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def run_experiments(budget_sec, experiment_time_override=None):
    experiments = read_todo()
    if not experiments:
        log("todo.log is empty — skipping experiment phase")
        return

    if experiment_time_override is not None:
        log(f"Experiment phase: {len(experiments)} queued, {budget_sec/60:.0f} min budget "
            f"(cutoff override: {experiment_time_override} min)")
    else:
        log(f"Experiment phase: {len(experiments)} queued, {budget_sec/60:.0f} min budget")
    clear_todo()

    phase_start = time.time()

    for i, exp in enumerate(experiments):
        elapsed   = time.time() - phase_start
        remaining = budget_sec - elapsed

        exp_id = exp.get("exp_id", f"exp_{int(time.time())}")

        # Need at least 90 s to do anything useful
        if remaining < 90:
            log(f"Only {remaining:.0f}s left — skipping {exp_id} and all remaining")
            break

        # Determine cutoff: override > experiment's own value, both capped to remaining budget
        requested_cutoff_min = (
            experiment_time_override
            if experiment_time_override is not None
            else float(exp.get("cutoff_minutes", 10))
        )
        if requested_cutoff_min < EXPERIMENT_MIN_MINUTES and experiment_time_override is None:
            log(f"{exp_id}: requested cutoff {requested_cutoff_min:.1f}min < minimum "
                f"{EXPERIMENT_MIN_MINUTES}min — clamping up")
            requested_cutoff_min = EXPERIMENT_MIN_MINUTES
        available_min        = (remaining - EXPERIMENT_GRACE_SECONDS) / 60
        capped_cutoff_min    = min(requested_cutoff_min, available_min)

        if capped_cutoff_min < 1.0:
            log(f"Not enough budget for {exp_id} (need >1 min, have {capped_cutoff_min:.1f}) — skipping")
            break

        log(
            f"[{i+1}/{len(experiments)}] Running {exp_id} "
            f"(requested={requested_cutoff_min:.1f}min, capped={capped_cutoff_min:.1f}min)"
        )

        cmd          = build_train_cmd(exp, exp_id, capped_cutoff_min)
        hard_timeout = capped_cutoff_min * 60 + EXPERIMENT_GRACE_SECONDS

        try:
            subprocess.run(cmd, timeout=hard_timeout)
        except subprocess.TimeoutExpired:
            log(f"{exp_id}: hard timeout ({hard_timeout:.0f}s) — process killed")
        except Exception as e:
            log(f"{exp_id}: subprocess error — {e}")

    log(f"Experiment phase done. Total elapsed: {(time.time()-phase_start)/60:.1f} min")


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    experiment_minutes = args.experiment_time or EXPERIMENT_MINUTES
    cycle_minutes      = RESEARCHER_MINUTES + experiment_minutes

    log("=" * 60)
    log("Auto-Researcher Scheduler started")
    log(f"  Researcher budget : {RESEARCHER_MINUTES} min / cycle")
    log(f"  Experiment budget : {experiment_minutes} min / cycle")
    log(f"  Cycle length      : {cycle_minutes} min")
    if args.experiment_time is not None:
        log(f"  Experiment cutoff override: {args.experiment_time} min per experiment")
    log("=" * 60)

    cycle = 0

    while True:
        cycle += 1
        cycle_start = time.time()
        log(f"=== Cycle {cycle} started ===")

        # Phase 1: Researcher
        run_researcher(RESEARCHER_MINUTES * 60)

        # Phase 2: Experiments — use whatever time is left in the cycle
        elapsed           = time.time() - cycle_start
        experiment_budget = max(0, cycle_minutes * 60 - elapsed)
        run_experiments(experiment_budget, args.experiment_time)

        total_elapsed = time.time() - cycle_start
        leftover      = cycle_minutes * 60 - total_elapsed
        if leftover > 0:
            log(f"Cycle {cycle} finished {leftover:.0f}s early — starting next cycle immediately")

        log(f"=== Cycle {cycle} complete ===\n")


if __name__ == "__main__":
    main()
