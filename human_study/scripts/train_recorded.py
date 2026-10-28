#!/usr/bin/env python3
"""Run native BC training with baseline settings and preserve verification records.

Activate the Isaac Lab environment first. The output directory and experiment
label must be new. Only the registered JSON's train.seed is changed temporarily;
its exact original bytes are restored, including after a failed training run.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

TASK = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"
CONFIG = "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/agents/robomimic/bc_rnn_low_dim.json"
EXCEPTIONS = {"train.data", "train.output_dir", "experiment.name"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fields(config, prefix=""):
    result = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            result.update(fields(value, name))
        else:
            result[name] = value
    return result


def compare(actual, baseline, *, subset=False):
    actual, baseline = fields(actual), fields(baseline)
    excluded = EXCEPTIONS | ({"train.seed"} if subset else set())
    keys = set(actual) if subset else set(actual) | set(baseline)
    different = [key for key in sorted(keys - excluded)
                 if key not in actual or key not in baseline or actual[key] != baseline[key]]
    if different:
        raise ValueError("Baseline configuration mismatch: " + ", ".join(different))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(101, 111), required=True)
    for key in ("dataset", "output-dir", "baseline-config"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lab-dir", type=Path, default=Path(__file__).resolve().parents[2] / "IsaacLab")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9_-]+", args.label) is None:
        raise ValueError("--label must contain only letters, digits, underscores, or hyphens")
    lab, dataset = args.lab_dir.resolve(strict=True), args.dataset.resolve(strict=True)
    baseline_path = args.baseline_config.resolve(strict=True)
    baseline = json.loads(baseline_path.read_text())
    if baseline["train"]["seed"] != args.seed or baseline["train"]["num_epochs"] != 2000:
        raise ValueError("Baseline must match the requested seed and have 2000 epochs")
    config_path = lab / CONFIG
    native_train = lab / "scripts/imitation_learning/robomimic/train.py"
    experiment_root = lab / "logs/human_study" / TASK
    experiment = experiment_root / args.label
    if experiment.exists():
        raise FileExistsError(f"Experiment already exists: {experiment}")
    output = args.output_dir.resolve()
    command = [str(lab / "isaaclab.sh"), "-p", "-u", str(native_train), "--task", TASK,
               "--algo", "bc", "--dataset", str(dataset), "--name", args.label, "--log_dir", "human_study"]
    with config_path.open("r+b") as locked_config:
        fcntl.flock(locked_config.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = locked_config.read()
        registered = json.loads(original)
        compare(registered, baseline, subset=True)
        output.mkdir(parents=True, exist_ok=False)
        metadata = {"status": "running", "label": args.label, "seed": args.seed, "task": TASK,
                    "started_at": datetime.now(timezone.utc).isoformat(), "command": command, "cwd": str(lab),
                    "dataset": str(dataset), "dataset_sha256": sha256(dataset),
                    "baseline_config": str(baseline_path), "baseline_config_sha256": sha256(baseline_path),
                    "native_train_sha256": sha256(native_train),
                    "registered_config_original_sha256": hashlib.sha256(original).hexdigest()}
        write_json(output / "run.json", metadata)
        (output / "registered_config_before.json").write_bytes(original)
        process, error, finished, native_error = None, None, False, False
        try:
            registered["train"]["seed"] = args.seed
            config_path.write_text(json.dumps(registered, indent=2) + "\n")
            with (output / "training.log").open("x", encoding="utf-8", buffering=1) as log:
                process = subprocess.Popen(command, cwd=lab, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace", start_new_session=True)
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                    finished |= line.strip() == "finished run successfully!"
                    native_error |= "run failed with error:" in line or "Traceback (most recent call last):" in line
                if process.wait() != 0 or not finished or native_error:
                    raise RuntimeError("Native training did not finish cleanly; inspect training.log")
            checkpoints = list(experiment.glob("*/models/model_epoch_2000.pth"))
            if len(checkpoints) != 1:
                raise RuntimeError(f"Expected one new final checkpoint, found {len(checkpoints)}")
            checkpoint = checkpoints[0]
            effective = json.loads((checkpoint.parent.parent / "config.json").read_text())
            compare(effective, baseline)
            if (effective["train"]["seed"] != args.seed or effective["train"]["data"] != str(dataset)
                    or effective["train"]["output_dir"] != str(experiment_root)
                    or effective["experiment"]["name"] != args.label):
                raise ValueError("Effective training seed, dataset, output directory, or label is incorrect")
            if sha256(dataset) != metadata["dataset_sha256"] or sha256(native_train) != metadata["native_train_sha256"]:
                raise RuntimeError("Dataset or native training source changed during training")
            write_json(output / "effective_config.json", effective)
            metadata.update(status="completed", checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                            final_config=effective, configuration_matches_baseline=True)
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            metadata.update(status="failed", error=error)
        finally:
            try:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            finally:
                config_path.write_bytes(original)
            metadata.update(returncode=process.returncode if process else None,
                            finished_at=datetime.now(timezone.utc).isoformat(),
                            registered_config_restored=config_path.read_bytes() == original)
            write_json(output / "summary.json", metadata)
    if error:
        print(f"Training failed: {error}; records: {output}", file=sys.stderr)
        return 1
    print(f"Verified final checkpoint: {metadata['checkpoint']}")
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
