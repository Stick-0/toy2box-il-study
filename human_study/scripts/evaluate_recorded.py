#!/usr/bin/env python3
"""Run the course evaluator unchanged and preserve validated per-trial results.

Activate the Isaac Lab conda environment before running. --output-dir must be
a new directory: an earlier evaluation is never overwritten.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


TASK = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"
ROLLOUTS = 100
HORIZON = 800
EVAL_SEED = 42
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRIAL = re.compile(r"^\[INFO\] Trial (\d+): (True|False)$")
COUNTS = re.compile(r"^Successful trials: (\d+), out of (\d+) trials$")
RATE = re.compile(r"^Success rate: ([0-9.eE+-]+)$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Results:
    """Collect only native evaluator result lines, retaining parse errors."""

    def __init__(self):
        self.trials = []
        self.counts = []
        self.rates = []

    def feed(self, line):
        line = ANSI.sub("", line).strip()
        match = TRIAL.fullmatch(line)
        if match:
            record = {"trial": int(match[1]), "success": match[2] == "True"}
            self.trials.append(record)
            return record
        match = COUNTS.fullmatch(line)
        if match:
            self.counts.append((int(match[1]), int(match[2])))
        match = RATE.fullmatch(line)
        if match:
            self.rates.append(float(match[1]))
        return None

    def validate(self):
        indices = [record["trial"] for record in self.trials]
        if indices != list(range(ROLLOUTS)):
            raise ValueError(f"Expected consecutive trials 0..99; observed {indices}")
        successful = sum(record["success"] for record in self.trials)
        rate = successful / ROLLOUTS
        if self.counts != [(successful, ROLLOUTS)]:
            raise ValueError(f"Missing or inconsistent native count summary: {self.counts}")
        if len(self.rates) != 1 or not math.isclose(self.rates[0], rate, abs_tol=1e-12):
            raise ValueError(f"Missing or inconsistent native success rate: {self.rates}")
        return successful, rate


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lab-dir", type=Path, default=Path(__file__).resolve().parents[2] / "IsaacLab")
    parser.add_argument("--video-folder", "--video_folder", type=Path, help="Optional video directory; omitted by default.")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    lab_dir = args.lab_dir.resolve(strict=True)
    play = (lab_dir / "scripts/imitation_learning/robomimic/play.py").resolve(strict=True)
    launcher = (lab_dir / "isaaclab.sh").resolve(strict=True)
    output = args.output_dir.resolve()
    command = [
        str(launcher), "-p", "-u", str(play),
        "--task", TASK, "--checkpoint", str(checkpoint),
        "--seed", str(EVAL_SEED), "--num_rollouts", str(ROLLOUTS),
        "--horizon", str(HORIZON), "--headless",
    ]
    if args.video_folder is not None:
        command += ["--video", "--video_length", "1000", "--video_folder", str(args.video_folder.resolve())]
    metadata = {
        "label": args.label, "status": "running", "started_at": utc_now(),
        "task": TASK, "evaluation_seed": EVAL_SEED, "num_rollouts": ROLLOUTS,
        "horizon": HORIZON, "video": args.video_folder is not None,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "native_play": str(play), "native_play_sha256": sha256(play),
        "command": command, "cwd": str(lab_dir),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", metadata)
    results = Results()
    process = None
    error = None
    try:
        with (output / "evaluation.log").open("x", encoding="utf-8", buffering=1) as log, \
                (output / "trials.jsonl").open("x", encoding="utf-8", buffering=1) as trials:
            process = subprocess.Popen(
                command, cwd=lab_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=True,
            )
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
                record = results.feed(line)
                if record is not None:
                    record = {**record, "recorded_at": utc_now()}
                    trials.write(json.dumps(record) + "\n")
                    trials.flush()
                    os.fsync(trials.fileno())
            returncode = process.wait()
            metadata["returncode"] = returncode
            if returncode != 0:
                raise RuntimeError(f"Native evaluator exited with status {returncode}")
            successful, rate = results.validate()
            if sha256(checkpoint) != metadata["checkpoint_sha256"] or sha256(play) != metadata["native_play_sha256"]:
                raise RuntimeError("Checkpoint or evaluator source changed during evaluation")
            metadata.update(status="completed", successful_trials=successful, success_rate=rate)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        metadata.update(status="failed", error=error)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process is not None:
            metadata["returncode"] = process.returncode
        metadata.update(finished_at=utc_now(), recorded_trials=len(results.trials))
        write_json(output / "summary.json", metadata)
    if error is not None:
        print(f"Evaluation failed: {error}\nPreserved records: {output}", file=sys.stderr)
        return 1
    print(f"Validated {metadata['successful_trials']}/{ROLLOUTS}; records: {output}")
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
