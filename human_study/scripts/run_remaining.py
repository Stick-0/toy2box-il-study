#!/usr/bin/env python3
"""Finish the fixed Mimic comparison after an existing generation process exits.

Inherit the activated Isaac Lab environment. Stops on the first error, never
overwrites stage outputs, and does not stop the externally started generator.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


SCRIPTS = Path(__file__).resolve().parent
DATA = SCRIPTS.parent / "data"


def process_matches(pid, text):
    try:
        return text.encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-pid", required=True, type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--generated", type=Path, default=DATA / "human30_mimic500_raw_20261028_env8.hdf5")
    parser.add_argument("--human", type=Path, default=DATA / "human_20261028_112300.hdf5")
    parser.add_argument("--baseline-pids", type=int, nargs=2, metavar=("SEED101_PID", "SEED111_PID"))
    parser.add_argument("--baseline-wait-timeout", type=float, help="Optional seconds to wait for baseline summaries.")
    args = parser.parse_args()
    if args.generation_pid <= 0 or (args.baseline_pids and any(pid <= 0 for pid in args.baseline_pids)):
        parser.error("Process IDs must be positive")
    if args.baseline_wait_timeout is not None and args.baseline_wait_timeout <= 0:
        parser.error("--baseline-wait-timeout must be positive")
    run_dir, generated, human = (path.resolve() for path in (args.run_dir, args.generated, args.human))
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "pipeline_status.json"
    if status_path.exists():
        raise FileExistsError(f"Refusing to overwrite {status_path}")
    children = []
    state = {"status": "running", "stage": "starting", "driver_pid": os.getpid(),
             "generation_pid": args.generation_pid, "generated": str(generated), "human": str(human),
             "started_at": datetime.now(timezone.utc).isoformat()}

    def status(stage=None, **values):
        if stage is not None:
            state["stage"] = stage
            print(f"[pipeline] {stage}", flush=True)
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat(),
                     child_pids=[child.pid for child in children if child.poll() is None])
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(status_path)

    def command(script, *arguments):
        return [sys.executable, "-u", str(SCRIPTS / script), *map(str, arguments)]

    with (run_dir / "pipeline.log").open("x", buffering=1) as log:
        def run(stage, commands):
            status(stage)
            for argv in commands:
                log.write("\n[pipeline command] " + json.dumps(argv) + "\n")
                log.flush()
                children.append(subprocess.Popen(argv, cwd=SCRIPTS.parent.parent, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True))
            status()
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError(f"{stage} failed; inspect pipeline.log and stage summaries")
                time.sleep(1)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError(f"{stage} failed; inspect pipeline.log and stage summaries")
            children.clear()

        try:
            status("waiting_for_generation")
            while process_matches(args.generation_pid, str(generated)):
                time.sleep(10)
                status()
            # No HDF5 file is opened until the recorded generation process ends.
            combined = generated.parent / "human30_plus_mimic500_20261028.hdf5"
            run("preparing_training_dataset", [command(
                "prepare_mimic_training.py", "--human", human, "--mimic", generated,
                "--output", combined, "--manifest", run_dir / "dataset_manifest.json", "--limit", 500)])
            for seed in (101, 111):
                label = f"mimic500_seed{seed}"
                run(f"training_{label}", [command(
                    "train_recorded.py", "--seed", seed, "--dataset", combined, "--label", label,
                    "--output-dir", run_dir / "training" / label,
                    "--baseline-config", run_dir / "configs" / f"baseline_seed{seed}.json")])
            evaluations = []
            for seed in (101, 111):
                label = f"mimic500_seed{seed}"
                summary = json.loads((run_dir / "training" / label / "summary.json").read_text())
                if summary.get("status") != "completed":
                    raise RuntimeError(f"Training is incomplete: {label}")
                evaluations.append(command("evaluate_recorded.py", "--checkpoint", summary["checkpoint"],
                                           "--label", label, "--output-dir", run_dir / "evaluation" / label))
            run("evaluating_mimic_models", evaluations)
            status("waiting_for_baseline_evaluations")
            wait_started = time.monotonic()
            while True:
                pending = []
                for index, seed in enumerate((101, 111)):
                    label = f"baseline_seed{seed}"
                    summary_path = run_dir / "evaluation" / label / "summary.json"
                    if summary_path.exists():
                        summary = json.loads(summary_path.read_text())
                        if summary.get("status") != "completed":
                            raise RuntimeError(f"Baseline evaluation failed or incomplete: {label}")
                    else:
                        pending.append(label)
                        if args.baseline_pids and not process_matches(args.baseline_pids[index], label):
                            raise RuntimeError(f"Baseline process ended without a summary: {label}")
                if not pending:
                    break
                if args.baseline_wait_timeout is not None and time.monotonic() - wait_started > args.baseline_wait_timeout:
                    raise TimeoutError(f"Configured baseline wait deadline exceeded: {pending}")
                status(pending_baselines=pending)
                time.sleep(10)
            run("summarizing_comparison", [command("summarize_comparison.py", "--run-dir", run_dir)])
            status("finished", status="completed", pending_baselines=[])
            return 0
        except (Exception, KeyboardInterrupt) as error:
            status(status="failed", error=f"{type(error).__name__}: {error}")
            print(f"[pipeline] failed: {error}", file=sys.stderr, flush=True)
            return 1
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            for child in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for child in children:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            status()


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
