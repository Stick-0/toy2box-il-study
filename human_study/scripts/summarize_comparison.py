#!/usr/bin/env python3
"""Validate four recorded evaluations and summarize the controlled comparison."""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tempfile


CONDITIONS = ("baseline", "mimic500")
SEEDS = (101, 111)
TASK = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_file(path_value, expected_hash, description):
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{description}: missing file path")
    path = Path(path_value).expanduser().resolve(strict=True)
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"{description}: missing or malformed SHA256")
    if sha256(path) != expected_hash:
        raise ValueError(f"{description}: file SHA256 differs from its recorded value")
    return path


def checkpoint_config(checkpoint, seed):
    # These are locally produced, hash-verified course checkpoints. Load on CPU;
    # inspecting metadata must not launch Isaac Sim or allocate a CUDA model.
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
    embedded = json.loads(saved) if isinstance(saved, str) else saved
    config_path = checkpoint.parent.parent / "config.json"
    config = json.loads(config_path.read_text())
    if embedded != config:
        raise ValueError(f"{checkpoint}: embedded config differs from saved config.json")
    if config["train"]["seed"] != seed or config["train"]["num_epochs"] != 2000:
        raise ValueError(f"{checkpoint}: expected training seed {seed} and 2000 epochs")
    if checkpoint.name != "model_epoch_2000.pth":
        raise ValueError(f"{checkpoint}: only the final epoch 2000 checkpoint is allowed")
    return config, config_path


def controlled_config(config, *, exclude_seed=False):
    result = json.loads(json.dumps(config))
    for key in ("data", "output_dir"):
        result["train"].pop(key, None)
    result["experiment"].pop("name", None)
    if exclude_seed:
        result["train"].pop("seed", None)
    return result


def recorded_hash(value, description):
    """Check a historical digest without claiming to read its original file."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{description}: missing or malformed SHA256")
    return value


def historical_path(value, description):
    """Compare original execution paths as records, without filesystem resolution."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description}: missing historical path")
    return Path(value)


def submission_checkpoints(run_dir, repo_root):
    """Validate portable records and the two submitted models, without baseline weights or HDF5."""
    audit_path = run_dir / "baseline_audit.json"
    runs = json.loads(audit_path.read_text()).get("runs")
    if not isinstance(runs, list) or len(runs) != len(SEEDS):
        raise ValueError("baseline_audit.json must contain exactly two baseline runs")
    by_seed = {}
    for run in runs:
        seed = run.get("training_seed")
        if type(seed) is not int or seed not in SEEDS or seed in by_seed:
            raise ValueError("baseline_audit.json must contain unique training seeds 101 and 111")
        if run.get("training_finished_successfully") is not True:
            raise ValueError(f"Baseline seed {seed} has no successful training record")
        by_seed[seed] = run

    manifest_path = run_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    execution = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("schema") != "toy2box.mimic_training_dataset.v1":
        raise ValueError("Unexpected dataset manifest schema")
    dataset = manifest["output"]
    human = manifest["sources"]["human"]
    mimic = manifest["sources"]["mimic"]
    for field, value in {"human_count": 30, "mimic_count": 500, "demo_count": 530, "env_name": TASK}.items():
        if dataset.get(field) != value or execution["training_dataset"].get(field) != value:
            raise ValueError(f"Dataset records {field} must be {value!r}")
    for description, record in (("Human source", human), ("Mimic source", mimic), ("Training dataset", dataset)):
        recorded_hash(record.get("sha256"), description)
        historical_path(record.get("path"), description)
    if (human.get("demo_count") != 30 or mimic.get("demo_count") != 500
            or mimic.get("successful_count") != 500 or manifest.get("limit") != 500
            or execution.get("human_dataset_sha256") != human["sha256"]
            or execution["training_dataset"].get("sha256") != dataset["sha256"]):
        raise ValueError("Recorded dataset counts or SHA256 evidence disagree")
    if (historical_path(execution.get("human_dataset"), "Execution human dataset") != Path(human["path"])
            or historical_path(execution["training_dataset"].get("path"), "Execution training dataset")
            != Path(dataset["path"])):
        raise ValueError("Recorded dataset paths disagree")

    expected = {}
    baseline_configs = {}
    native_train_hash = sha256(repo_root / "IsaacLab/scripts/imitation_learning/robomimic/train.py")
    for seed in SEEDS:
        baseline_record = by_seed[seed]
        baseline_path = run_dir / "configs" / f"baseline_seed{seed}.json"
        baseline = json.loads(baseline_path.read_text())
        if baseline["train"]["seed"] != seed or baseline["train"]["num_epochs"] != 2000:
            raise ValueError(f"Baseline seed {seed}: expected its training seed and 2000 epochs")
        baseline_checkpoint = historical_path(baseline_record.get("checkpoint_path"), "Baseline checkpoint")
        baseline_hash = recorded_hash(baseline_record.get("checkpoint_sha256"), "Baseline checkpoint")
        if (baseline_checkpoint.name != "model_epoch_2000.pth"
                or historical_path(baseline_record.get("config_path"), "Baseline config")
                != baseline_checkpoint.parent.parent / "config.json"):
            raise ValueError(f"Baseline seed {seed}: inconsistent historical checkpoint/config paths")
        if historical_path(baseline["train"].get("data"), "Baseline dataset") != Path(human["path"]):
            raise ValueError(f"Baseline seed {seed}: recorded training data is not the human source")
        baseline_configs[seed] = baseline
        expected["baseline", seed] = {
            "checkpoint": str(baseline_checkpoint), "recorded_checkpoint_path": str(baseline_checkpoint),
            "checkpoint_sha256": baseline_hash, "checkpoint_file_verified": False,
            "embedded_config_verified": False, "config_sha256": sha256(baseline_path),
            "training_record_sha256": sha256(audit_path), "training_dataset_sha256": human["sha256"],
            "training_dataset_file_verified": False,
        }

        label = f"mimic500_seed{seed}"
        training_path = run_dir / "training" / label / "summary.json"
        training = json.loads(training_path.read_text())
        for field, value in {"status": "completed", "label": label, "seed": seed, "task": TASK,
                             "returncode": 0, "configuration_matches_baseline": True,
                             "registered_config_restored": True}.items():
            if training.get(field) != value:
                raise ValueError(f"{label} training record: {field} must be {value!r}")
        if training.get("baseline_config_sha256") != sha256(baseline_path):
            raise ValueError(f"{label}: preserved baseline config hash differs from training record")
        if training.get("native_train_sha256") != native_train_hash:
            raise ValueError(f"{label}: submitted training source differs from recorded source")
        original_checkpoint = historical_path(training.get("checkpoint"), f"{label} checkpoint")
        if original_checkpoint.name != "model_epoch_2000.pth":
            raise ValueError(f"{label}: only the recorded final epoch 2000 model is allowed")
        checkpoint_relative = Path("models") / f"{label}.pth"
        checkpoint = recorded_file(str(repo_root / checkpoint_relative), training.get("checkpoint_sha256"), label)
        import torch

        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
        config = json.loads(saved) if isinstance(saved, str) else saved
        config_path = training_path.parent / "effective_config.json"
        if config != training.get("final_config") or config != json.loads(config_path.read_text()):
            raise ValueError(f"{label}: embedded config differs from preserved training records")
        if config["train"]["seed"] != seed or config["train"]["num_epochs"] != 2000:
            raise ValueError(f"{label}: expected its training seed and 2000 epochs")
        if controlled_config(config) != controlled_config(baseline):
            raise ValueError(f"{label}: training settings differ from baseline beyond data/output_dir/name")
        if (historical_path(training.get("dataset"), f"{label} dataset") != Path(dataset["path"])
                or historical_path(config["train"].get("data"), f"{label} config dataset") != Path(dataset["path"])
                or training.get("dataset_sha256") != dataset["sha256"]):
            raise ValueError(f"{label}: recorded dataset path/hash does not match dataset manifest")
        expected["mimic500", seed] = {
            "checkpoint": checkpoint_relative.as_posix(), "recorded_checkpoint_path": str(original_checkpoint),
            "checkpoint_sha256": training["checkpoint_sha256"], "checkpoint_file_verified": True,
            "embedded_config_verified": True, "config_sha256": sha256(config_path),
            "training_record_sha256": sha256(training_path), "training_dataset_sha256": dataset["sha256"],
            "training_dataset_file_verified": False, "dataset_manifest_sha256": sha256(manifest_path),
        }
    if controlled_config(baseline_configs[101], exclude_seed=True) != controlled_config(
            baseline_configs[111], exclude_seed=True):
        raise ValueError("The two baseline training settings differ beyond seed/data/output_dir/name")
    if len({record["recorded_checkpoint_path"] for record in expected.values()}) != 4:
        raise ValueError("Each condition and seed must have its own recorded checkpoint")
    return expected


def verify_native_log(directory, trials, successful, rate):
    """Cross-check JSON outcomes against the retained native evaluator output."""
    path = directory / "evaluation.log"
    text = path.read_text()
    logged = [(int(index), outcome == "True") for index, outcome in
              re.findall(r"^\[INFO\] Trial (\d+): (True|False)$", text, re.MULTILINE)]
    if logged != [(trial["trial"], trial["success"]) for trial in trials]:
        raise ValueError(f"{directory.name}: native log trial outcomes disagree with trials.jsonl")
    counts = re.findall(r"^Successful trials: (\d+), out of (\d+) trials$", text, re.MULTILINE)
    rates = re.findall(r"^Success rate: ([0-9.eE+-]+)$", text, re.MULTILINE)
    outcomes = re.findall(r"^Trial Results: (\[.*\])$", text, re.MULTILINE)
    if (counts != [(str(successful), "100")] or len(rates) != 1
            or not math.isclose(float(rates[0]), rate, abs_tol=1e-12)
            or len(outcomes) != 1 or ast.literal_eval(outcomes[0]) != [trial["success"] for trial in trials]):
        raise ValueError(f"{directory.name}: native log final counts or outcomes disagree with JSON records")
    return sha256(path)


def expected_checkpoints(run_dir):
    """Bind condition labels to training records, actual files, and embedded configs."""
    audit_path = run_dir / "baseline_audit.json"
    audit = json.loads(audit_path.read_text())
    runs = audit.get("runs")
    if not isinstance(runs, list) or len(runs) != len(SEEDS):
        raise ValueError("baseline_audit.json must contain exactly two baseline runs")
    by_seed = {}
    for run in runs:
        seed = run.get("training_seed")
        if type(seed) is not int or seed not in SEEDS or seed in by_seed:
            raise ValueError("baseline_audit.json must contain unique training seeds 101 and 111")
        if run.get("training_finished_successfully") is not True:
            raise ValueError(f"Baseline seed {seed} has no successful training record")
        by_seed[seed] = run

    manifest_path = run_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "toy2box.mimic_training_dataset.v1":
        raise ValueError("Unexpected dataset manifest schema")
    dataset_record = manifest["output"]
    for field, value in {"human_count": 30, "mimic_count": 500, "demo_count": 530, "env_name": TASK}.items():
        if dataset_record.get(field) != value:
            raise ValueError(f"Dataset manifest {field} must be {value!r}")
    dataset = recorded_file(dataset_record.get("path"), dataset_record.get("sha256"), "Mimic training dataset")
    human_record = manifest["sources"]["human"]
    human = recorded_file(human_record.get("path"), human_record.get("sha256"), "Human source dataset")

    expected = {}
    baseline_configs = {}
    for seed in SEEDS:
        baseline_record = by_seed[seed]
        baseline_checkpoint = recorded_file(baseline_record.get("checkpoint_path"),
                                            baseline_record.get("checkpoint_sha256"), f"Baseline seed {seed}")
        baseline, baseline_config_path = checkpoint_config(baseline_checkpoint, seed)
        if Path(baseline_record["config_path"]).resolve(strict=True) != baseline_config_path.resolve(strict=True):
            raise ValueError(f"Baseline seed {seed}: audit config path does not belong to its checkpoint")
        if Path(baseline["train"]["data"]).resolve(strict=True) != human:
            raise ValueError(f"Baseline seed {seed}: training data is not the human source in dataset_manifest.json")
        baseline_configs[seed] = baseline
        expected["baseline", seed] = {
            "checkpoint": str(baseline_checkpoint), "checkpoint_sha256": baseline_record["checkpoint_sha256"],
            "config_sha256": sha256(baseline_config_path), "training_record_sha256": sha256(audit_path),
            "training_dataset_sha256": human_record["sha256"],
        }

        label = f"mimic500_seed{seed}"
        training_path = run_dir / "training" / label / "summary.json"
        training = json.loads(training_path.read_text())
        for field, value in {"status": "completed", "label": label, "seed": seed, "task": TASK,
                             "returncode": 0, "configuration_matches_baseline": True,
                             "registered_config_restored": True}.items():
            if training.get(field) != value:
                raise ValueError(f"{label} training record: {field} must be {value!r}")
        checkpoint = recorded_file(training.get("checkpoint"), training.get("checkpoint_sha256"), label)
        config, config_path = checkpoint_config(checkpoint, seed)
        if config != training.get("final_config"):
            raise ValueError(f"{label}: saved config differs from the completed training record")
        if controlled_config(config) != controlled_config(baseline):
            raise ValueError(f"{label}: training settings differ from baseline beyond data/output_dir/name")
        reference = recorded_file(training.get("baseline_config"), training.get("baseline_config_sha256"),
                                  f"{label} baseline config reference")
        if json.loads(reference.read_text()) != baseline:
            raise ValueError(f"{label}: training baseline config reference is not the audited baseline")
        if (Path(training["dataset"]).resolve(strict=True) != dataset
                or Path(config["train"]["data"]).resolve(strict=True) != dataset
                or training.get("dataset_sha256") != dataset_record["sha256"]):
            raise ValueError(f"{label}: training dataset path/hash does not match dataset_manifest.json")
        expected["mimic500", seed] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": training["checkpoint_sha256"],
            "config_sha256": sha256(config_path), "training_record_sha256": sha256(training_path),
            "training_dataset_sha256": dataset_record["sha256"],
            "dataset_manifest_sha256": sha256(manifest_path),
        }
    if controlled_config(baseline_configs[101], exclude_seed=True) != controlled_config(
            baseline_configs[111], exclude_seed=True):
        raise ValueError("The two baseline training settings differ beyond seed/data/output_dir/name")
    if len({item["checkpoint"] for item in expected.values()}) != 4:
        raise ValueError("Each condition and seed must refer to its own final checkpoint")
    return expected


def load_evaluations(run_dir, *, submission=False, repo_root=None):
    if submission:
        repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve(strict=True)
        models = submission_checkpoints(run_dir, repo_root)
        submitted_play_hash = sha256(repo_root / "IsaacLab/scripts/imitation_learning/robomimic/play.py")
    else:
        models = expected_checkpoints(run_dir)
    records = []
    signatures = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            label = f"{condition}_seed{seed}"
            directory = run_dir / "evaluation" / label
            summary_path = directory / "summary.json"
            trials_path = directory / "trials.jsonl"
            summary = json.loads(summary_path.read_text())
            expected = {
                "label": label, "status": "completed", "task": TASK,
                "num_rollouts": 100, "recorded_trials": 100,
                "evaluation_seed": 42, "horizon": 800, "returncode": 0,
            }
            for key, value in expected.items():
                if summary.get(key) != value:
                    raise ValueError(f"{label}: {key} must be {value!r}, got {summary.get(key)!r}")
            model = models[condition, seed]
            recorded_checkpoint = historical_path(summary.get("checkpoint"), f"{label} evaluated checkpoint")
            expected_checkpoint = Path(model.get("recorded_checkpoint_path", model["checkpoint"]))
            if ((recorded_checkpoint if submission else recorded_checkpoint.resolve(strict=True)) != expected_checkpoint
                    or summary.get("checkpoint_sha256") != model["checkpoint_sha256"]):
                raise ValueError(f"{label}: evaluated checkpoint does not match the verified training record")
            evaluator_hash = summary.get("native_play_sha256", "")
            if not isinstance(evaluator_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", evaluator_hash):
                raise ValueError(f"{label}: missing or malformed native evaluator hash")
            if submission and evaluator_hash != submitted_play_hash:
                raise ValueError(f"{label}: submitted native evaluator differs from the recorded evaluator")
            if type(summary.get("video")) is not bool:
                raise ValueError(f"{label}: video flag must be an explicit boolean")
            signatures.add((evaluator_hash, summary["video"]))
            trials = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]
            if len(trials) != 100 or any(
                type(trial.get("trial")) is not int or trial["trial"] != index
                or type(trial.get("success")) is not bool
                for index, trial in enumerate(trials)
            ):
                raise ValueError(f"{label}: expected exactly trials 0..99 with boolean outcomes")
            successful = sum(trial["success"] for trial in trials)
            rate = successful / 100
            if summary.get("successful_trials") != successful or not math.isclose(
                float(summary.get("success_rate", float("nan"))), rate, abs_tol=1e-12
            ):
                raise ValueError(f"{label}: trial outcomes disagree with summary")
            log_evidence = {}
            if submission:
                command = summary.get("command")
                if not isinstance(command, list):
                    raise ValueError(f"{label}: missing recorded evaluation command")
                for flag, value in {"--task": TASK, "--checkpoint": summary["checkpoint"],
                                    "--seed": "42", "--num_rollouts": "100", "--horizon": "800"}.items():
                    if (command.count(flag) != 1 or command.index(flag) + 1 >= len(command)
                            or command[command.index(flag) + 1] != value):
                        raise ValueError(f"{label}: evaluation command disagrees with {flag}")
                if ("--video" in command) != summary["video"]:
                    raise ValueError(f"{label}: evaluation command disagrees with the video flag")
                log_evidence = {"native_log_sha256": verify_native_log(directory, trials, successful, rate),
                                "native_log_verified": True, "training_records_verified": True}
            records.append({
                "condition": condition, "training_seed": seed, "label": label,
                "successful_trials": successful, "num_rollouts": 100,
                "success_rate": rate, "success_percent": 100 * rate,
                **model,
                "training_provenance_verified": not submission,
                **log_evidence,
                "summary_sha256": sha256(summary_path), "trials_sha256": sha256(trials_path),
            })
    if len(signatures) != 1:
        raise ValueError("All four evaluations must use identical evaluator source and video settings")
    evaluator_hash, video = signatures.pop()
    return records, evaluator_hash, video


def compare(records, evaluator_hash, video):
    groups = {}
    for condition in CONDITIONS:
        rates = [record["success_percent"] for record in records if record["condition"] == condition]
        groups[condition] = {
            "training_seeds": list(SEEDS), "mean_success_percent": statistics.mean(rates),
            "sample_sd_percentage_points": statistics.stdev(rates), "sd_ddof": 1,
        }
    by_key = {(record["condition"], record["training_seed"]): record for record in records}
    paired = [{
        "training_seed": seed,
        "delta_percentage_points": (
            by_key["mimic500", seed]["success_percent"] - by_key["baseline", seed]["success_percent"]
        ),
    } for seed in SEEDS]
    deltas = [pair["delta_percentage_points"] for pair in paired]
    if all(delta > 0 for delta in deltas):
        verdict = "both_improve"
        interpretation = "Mimic improves success for both training seeds in this evaluation."
    elif any(delta > 0 for delta in deltas):
        verdict = "mixed"
        interpretation = "Mimic improves one training seed; the other is unchanged or worse."
    else:
        verdict = "no_improvement"
        interpretation = "Neither training seed improves with Mimic in this evaluation."
    mean_delta = statistics.mean(deltas)
    if mean_delta <= 0:
        support = "no_support"
        support_interpretation = "Mean success does not increase; these results do not support the predicted improvement."
    elif all(delta > 0 for delta in deltas):
        support = "consistent_improvement"
        support_interpretation = "Both seeds improve, consistently supporting the augmentation prediction in this evaluation."
    else:
        support = "limited_support"
        support_interpretation = "Mean success increases but improvement is not consistent across seeds; support is limited."
    return {
        "protocol": {"task": TASK, "evaluation_seed": 42, "rollouts_per_model": 100,
                     "horizon": 800, "native_play_sha256": evaluator_hash, "video": video},
        "models": records, "conditions": groups, "paired_by_training_seed": paired,
        "mean_delta_percentage_points": mean_delta,
        "pattern_verdict": verdict, "interpretation": interpretation,
        "hypothesis_support": support, "hypothesis_support_interpretation": support_interpretation,
        "limitations": [
            "Mean and sample SD (ddof=1) summarize only two training seeds; this is not a confidence interval.",
            "Deltas pair models by training seed, not individual rollout trajectories.",
            "Only evaluation seed 42 was tested; improvement on the undisclosed final seed is unknown.",
            "These descriptive results alone do not establish statistical significance or diagnose failure causes.",
        ],
    }


def save_artifacts(run_dir, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=run_dir) as temporary:
        staging = Path(temporary)
        (staging / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        columns = ["condition", "training_seed", "successful_trials", "num_rollouts", "success_percent",
                   "condition_mean_percent", "condition_sample_sd_percentage_points"]
        with (staging / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in comparison["models"]:
                group = comparison["conditions"][record["condition"]]
                row = {key: record[key] for key in columns[:5]}
                row.update(condition_mean_percent=group["mean_success_percent"],
                           condition_sample_sd_percentage_points=group["sample_sd_percentage_points"])
                writer.writerow(row)
        figure, axis = plt.subplots(figsize=(7, 4.8), layout="constrained")
        means = [comparison["conditions"][condition]["mean_success_percent"] for condition in CONDITIONS]
        deviations = [comparison["conditions"][condition]["sample_sd_percentage_points"] for condition in CONDITIONS]
        axis.bar([0, 1], means, yerr=deviations, capsize=6, width=0.55,
                 color=["#b8c9d9", "#98cbb2"], edgecolor="#415365", label="Mean ± sample SD")
        for seed, marker, color in zip(SEEDS, ("o", "D"), ("#173f73", "#b94c17")):
            values = [next(record["success_percent"] for record in comparison["models"]
                           if record["condition"] == condition and record["training_seed"] == seed)
                      for condition in CONDITIONS]
            axis.plot([0, 1], values, marker=marker, color=color, linewidth=1.2,
                      markersize=6, label=f"Training seed {seed}")
        for x, mean, deviation in zip((0, 1), means, deviations):
            axis.text(x, mean + deviation + 2.5, f"{mean:.1f} ± {deviation:.1f}%", ha="center", fontsize=10,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 2})
        axis.set_xticks([0, 1], ["Baseline", "Mimic 500"])
        axis.set_ylabel("Success rate (%)")
        axis.set_ylim(min(-3, min(m - s for m, s in zip(means, deviations)) - 5),
                      max(105, max(m + s for m, s in zip(means, deviations)) + 10))
        axis.set_title("Human demonstrations vs. Mimic augmentation\nEvaluation seed 42 · 100 rollouts/model · horizon 800")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", fontsize=9)
        figure.savefig(staging / "success_rates.png", dpi=180)
        plt.close(figure)
        (run_dir / "figures").mkdir(exist_ok=True)
        (staging / "success_rates.png").replace(run_dir / "figures" / "success_rates.png")
        for name in ("comparison.json", "comparison.csv"):
            (staging / name).replace(run_dir / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--submission", action="store_true", help=(
        "Verify preserved evaluation records and the two submitted models. "
        "Does not require or re-hash removed baseline weights or local HDF5 datasets."))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Submission repository root (defaults to the directory containing human_study).")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    records, evaluator_hash, video = load_evaluations(run_dir, submission=args.submission, repo_root=args.repo_root)
    comparison = compare(records, evaluator_hash, video)
    if args.submission:
        comparison["validation"] = {
            "mode": "submission",
            "evaluation_json_and_native_logs_verified": True,
            "submitted_native_evaluator_verified": True,
            "final_model_files_and_embedded_configs_verified": True,
            "baseline_checkpoint_files_verified": False,
            "dataset_files_verified": False,
            "dataset_evidence": "Cross-checked preserved manifest paths, counts, and SHA256 records only.",
        }
        comparison["limitations"].extend([
            "Submission validation uses preserved baseline configs, audit, evaluation summaries, 100 trial records "
            "per model, and native logs. Removed baseline weights and their embedded configs are not re-verified.",
            "HDF5 datasets are not opened or re-hashed in submission mode; their original counts and hashes are "
            "cross-checked as historical records. This does not establish current dataset file integrity.",
        ])
    save_artifacts(run_dir, comparison)
    for record in records:
        print(f"{record['label']}: {record['successful_trials']}/100 ({record['success_percent']:.1f}%)")
    for condition, group in comparison["conditions"].items():
        print(f"{condition}: {group['mean_success_percent']:.2f} ± "
              f"{group['sample_sd_percentage_points']:.2f}% (sample SD, n=2)")
    print(f"Mean delta: {comparison['mean_delta_percentage_points']:+.2f} percentage points")
    print(comparison["interpretation"])
    print(comparison["hypothesis_support_interpretation"])
    print(f"Saved comparison.json, comparison.csv, and figures/success_rates.png in {run_dir}")


if __name__ == "__main__":
    main()
