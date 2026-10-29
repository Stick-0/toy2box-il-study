#!/usr/bin/env python3
"""Run unchanged native Toy2Box training using a process-local JSON entry point.

The original configuration and native sources are read-only. This wrapper copies
the JSON into a new record directory, changes only train.seed, and replaces the
Gym registry configuration path in the child process. Native main(args) performs
all model construction, data loading, updates, and checkpoint saving.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys


TASK = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"
ENTRY_POINT = "robomimic_bc_cfg_entry_point"
EXCLUDED_FIELDS = {"train.data", "train.output_dir", "experiment.name"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def flatten(config, prefix=""):
    result = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def compare_config(actual, baseline, *, subset=False):
    # robomimic defaults contain tuples; saved JSON uses lists. Compare the
    # serialized configuration semantics rather than those container types.
    actual = json.loads(json.dumps(actual))
    baseline = json.loads(json.dumps(baseline))
    actual, baseline = flatten(actual), flatten(baseline)
    keys = set(actual) if subset else set(actual) | set(baseline)
    differences = [key for key in sorted(keys - EXCLUDED_FIELDS)
                   if key not in actual or key not in baseline or actual[key] != baseline[key]]
    if differences:
        raise ValueError("Baseline configuration mismatch: " + ", ".join(differences))


def native_child(spec_path):
    """Only this function launches Isaac Sim; the parent remains lightweight."""
    spec = json.loads(Path(spec_path).read_text())
    native = Path(spec["native_source"])
    if sha256(native) != spec["native_source_sha256"]:
        raise RuntimeError("Native source changed before child launch")
    config_path = Path(spec["input_config"])
    if sha256(config_path) != spec["input_config_sha256"]:
        raise RuntimeError("Local input configuration changed before child launch")
    # run_name deliberately differs from __main__: imports and AppLauncher run,
    # but the original CLI/main guard does not. The source bytes stay unchanged.
    sys.argv = [str(native)]
    namespace = runpy.run_path(str(native), run_name="toy2box_native_training")
    application = namespace["simulation_app"]
    try:
        # Check the same default expansion used by native main before any
        # training occurs, then let main independently load the local JSON.
        external_config = json.loads(config_path.read_text())
        expanded_config = namespace["config_factory"](external_config["algo_name"])
        with expanded_config.values_unlocked():
            expanded_config.update(external_config)
        compare_config(expanded_config, json.loads(Path(spec["baseline_config"]).read_text()))
        environment_spec = namespace["gym"].spec(TASK)
        if ENTRY_POINT not in environment_spec.kwargs:
            raise KeyError(f"Native task lacks {ENTRY_POINT}")
        environment_spec.kwargs[ENTRY_POINT] = str(config_path)
        args = argparse.Namespace(
            task=TASK, algo="bc", dataset=spec["dataset"], name=spec["label"],
            log_dir=spec["log_dir"], normalize_training_actions=False, epochs=None,
        )
        namespace["main"](args)
    finally:
        application.close()
    return 0


def validate(args):
    if re.fullmatch(r"[A-Za-z0-9_-]+", args.name) is None:
        raise ValueError("--name must contain only letters, digits, underscores, or hyphens")
    lab = args.lab_dir.expanduser().resolve(strict=True)
    native = args.native_source.expanduser().resolve(strict=True)
    dataset = args.dataset.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    baseline_path = args.baseline_config.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    experiment_root = log_dir / TASK
    experiment = experiment_root / args.name
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Record directory already exists: {output}")
    if experiment.exists() or experiment.is_symlink():
        raise FileExistsError(f"Native experiment already exists: {experiment}")
    # Confirm the native entry function's calling convention without importing
    # Isaac Sim during --validate-only.
    tree = ast.parse(native.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    if len(functions) != 1 or [arg.arg for arg in functions[0].args.args] != ["args"]:
        raise ValueError("Unexpected native main(args) signature")
    registered = json.loads(config_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    local_config = json.loads(json.dumps(registered))
    local_config["train"]["seed"] = args.seed
    required = {
        "train.seed": args.seed,
        "train.num_epochs": 2000,
        "train.batch_size": 100,
        "train.num_data_workers": 4,
        "experiment.epoch_every_n_steps": 100,
        "experiment.rollout.enabled": False,
        "experiment.validate": False,
    }
    for name, values in (("local", flatten(local_config)), ("baseline", flatten(baseline))):
        for key, expected in required.items():
            if values.get(key) != expected:
                raise ValueError(f"{name} {key} must be {expected!r}; got {values.get(key)!r}")
    compare_config(local_config, baseline, subset=True)
    return dict(lab=lab, native=native, dataset=dataset, config_path=config_path,
                baseline_path=baseline_path, output=output, log_dir=log_dir,
                experiment_root=experiment_root, experiment=experiment,
                baseline=baseline, local_config=local_config)


def run(args):
    context = validate(args)
    if args.validate_only:
        print(json.dumps({"status": "preflight_passed", "seed": args.seed,
                          "dataset": str(context["dataset"]),
                          "experiment": str(context["experiment"]),
                          "output_dir": str(context["output"]),
                          "configuration_check": "Supplied JSON fields match baseline; full default expansion is checked in the child before training and after native training",
                          "gpu_launched": False, "files_written": False}, indent=2))
        return 0
    output = context["output"]
    output.mkdir(parents=True, exist_ok=False)
    config_copy = output / "input_config.json"
    write_json(config_copy, context["local_config"])
    metadata = {
        "status": "running", "started_at": now(), "label": args.name,
        "seed": args.seed, "task": TASK, "cwd": str(context["lab"]),
        "dataset": str(context["dataset"]), "dataset_sha256": sha256(context["dataset"]),
        "config_source": str(context["config_path"]), "config_source_sha256": sha256(context["config_path"]),
        "input_config": str(config_copy), "input_config_sha256": sha256(config_copy),
        "baseline_config": str(context["baseline_path"]), "baseline_config_sha256": sha256(context["baseline_path"]),
        "native_source": str(context["native"]), "native_source_sha256": sha256(context["native"]),
        "wrapper": str(Path(__file__).resolve()), "wrapper_sha256": sha256(__file__),
        "log_dir": str(context["log_dir"]), "experiment_directory": str(context["experiment"]),
        "checkpoint_selection": "Fixed final epoch 2000; no evaluation-based selection",
        "configuration_override": "train.seed in local JSON; Gym entry point changed only in child memory",
        "environment": {key: os.environ.get(key) for key in
                        ("CONDA_PREFIX", "LD_PRELOAD", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    spec = output / "run.json"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--native-child", str(spec)]
    metadata["command"] = command
    write_json(spec, metadata)
    process, error, finished, native_error = None, None, False, False
    try:
        with (output / "training.log").open("x", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(command, cwd=context["lab"], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                       errors="replace", bufsize=1, start_new_session=True)
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
                finished |= line.strip() == "finished run successfully!"
                native_error |= "run failed with error:" in line or "Traceback (most recent call last):" in line
            if process.wait() != 0 or not finished or native_error:
                raise RuntimeError("Native training did not finish cleanly; inspect training.log")
        checkpoints = list(context["experiment"].glob("*/models/model_epoch_2000.pth"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"Expected exactly one final checkpoint, found {len(checkpoints)}")
        checkpoint = checkpoints[0]
        effective = json.loads((checkpoint.parent.parent / "config.json").read_text())
        compare_config(effective, context["baseline"])
        if (effective["train"]["seed"] != args.seed
                or effective["train"]["data"] != str(context["dataset"])
                or effective["train"]["output_dir"] != str(context["experiment_root"])
                or effective["experiment"]["name"] != args.name):
            raise ValueError("Effective seed, data path, output directory, or name is incorrect")
        for path, key in ((context["dataset"], "dataset_sha256"),
                          (context["native"], "native_source_sha256"),
                          (context["config_path"], "config_source_sha256"),
                          (context["baseline_path"], "baseline_config_sha256"),
                          (config_copy, "input_config_sha256"),
                          (Path(__file__), "wrapper_sha256")):
            if sha256(path) != metadata[key]:
                raise RuntimeError(f"Source or input changed during training: {path}")
        write_json(output / "effective_config.json", effective)
        metadata.update(status="completed", checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                        configuration_matches_baseline=True,
                        source_configuration_unchanged=True, final_config=effective)
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
        metadata.update(finished_at=now(), returncode=process.returncode if process else None)
        write_json(output / "summary.json", metadata)
    if error:
        print(f"Training failed: {error}; preserved records: {output}", file=sys.stderr)
        return 1
    print(f"Verified final checkpoint: {metadata['checkpoint']}")
    return 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--native-child":
        return native_child(sys.argv[2])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(101, 111), required=True)
    for name in ("config", "baseline-config", "native-source", "lab-dir", "dataset", "log-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--validate-only", action="store_true", help="Read-only preflight; no simulator or GPU launch")
    return run(parser.parse_args())


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
