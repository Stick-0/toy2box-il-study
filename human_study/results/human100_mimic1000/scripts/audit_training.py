#!/usr/bin/env python3
"""Read-only training/checkpoint audit; only the audit reports are written.

--complete requires both completed runs and CPU-loads their final checkpoints.
Without it, inspect a snapshot of the running training logs without importing
torch or touching a training process.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import h5py

ROOT = Path(__file__).resolve().parent
EXCEPTIONS = {"train.data", "train.output_dir", "experiment.name"}
EXPECTED_SHAPES = {"box_orientations": [4], "box_positions": [3], "eef_pos": [3],
                   "eef_quat": [4], "gripper_pos": [2], "toy_truck_orientations": [4],
                   "toy_truck_positions": [3]}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def flat(value, prefix=""):
    result = {}
    for key, entry in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(entry, dict) and entry:
            result.update(flat(entry, name))
        else:
            result[name] = entry
    return result


def compare(actual, expected, exclude=()):
    actual, expected = flat(actual), flat(expected)
    differences = [key for key in sorted((set(actual) | set(expected)) - set(exclude))
                   if key not in actual or key not in expected or actual[key] != expected[key]]
    if differences:
        raise ValueError("Configuration mismatch: " + ", ".join(differences))


def read_epochs(path, complete):
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    matches = list(re.finditer(r"^Train Epoch (\d+)\s*\n(\{.*?^\})", text, re.M | re.S))
    entries = [(int(match[1]), json.loads(match[2])) for match in matches]
    indices = [number for number, _ in entries]
    assert indices and indices == list(range(1, len(indices) + 1)), indices
    assert len(indices) <= 2000
    for number, metrics in entries:
        assert "Loss" in metrics and "Log_Likelihood" in metrics
        assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values()), number
        assert math.isclose(metrics["Loss"], -metrics["Log_Likelihood"], rel_tol=1e-8, abs_tol=1e-8), number
    errors = [line for line in text.splitlines()
              if "run failed with error:" in line or "Traceback (most recent call last):" in line]
    assert not errors, errors
    markers = text.count("finished run successfully!")
    if complete:
        assert indices == list(range(1, 2001)) and markers == 1
    losses = [metrics["Loss"] for _, metrics in entries]
    return {"log_path": str(path), "snapshot_bytes": len(data),
            "snapshot_sha256": hashlib.sha256(data).hexdigest(),
            "completed_logged_epochs": len(indices), "epoch_order_consecutive": True,
            "all_logged_numeric_metrics_finite": True, "completion_markers": markers,
            "first_loss": losses[0], "last_loss": losses[-1],
            "minimum_loss": min(losses), "maximum_loss": max(losses),
            "first_100_loss_mean": sum(losses[:100]) / len(losses[:100]),
            "latest_100_loss_mean": sum(losses[-100:]) / len(losses[-100:]),
            "loss_definition": "Negative GMM log likelihood; negative values are expected and do not directly measure rollout success."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    dataset = Path(manifest["output"]["path"])
    dataset_sha = sha(dataset)
    assert dataset_sha == manifest["output"]["sha256"]
    assert manifest["output"]["human_count"] == 100 and manifest["output"]["mimic_count"] == 1000
    assert manifest["output"]["demo_count"] == 1100
    with h5py.File(dataset, "r") as handle:
        assert len(handle["data"]) == 1100
        assert all(bool(demo.attrs["success"]) for demo in handle["data"].values())
    result = {"audited_at": datetime.now(timezone.utc).isoformat(),
              "scope": "complete" if args.complete else "running_snapshot",
              "status": "passed", "gpu_execution": False,
              "dataset": str(dataset), "dataset_sha256": dataset_sha,
              "human_count": 100, "mimic_count": 1000, "demo_count": 1100,
              "all_training_demos_stored_successful": True, "runs": []}
    for seed in (101, 111):
        folder = ROOT / "training" / f"seed{seed}"
        run = json.loads((folder / "run.json").read_text())
        assert run["seed"] == seed and run["dataset"] == str(dataset) and run["dataset_sha256"] == dataset_sha
        baseline = json.loads(Path(run["baseline_config"]).read_text())
        native_configs = list(Path(run["experiment_directory"]).glob("*/config.json"))
        assert len(native_configs) == 1
        actual = json.loads(native_configs[0].read_text())
        compare(actual, baseline, EXCEPTIONS)
        assert actual["train"]["seed"] == seed
        assert actual["train"]["data"] == str(dataset)
        assert actual["train"]["num_epochs"] == 2000
        assert actual["experiment"]["epoch_every_n_steps"] == 100
        assert actual["train"]["batch_size"] == 100
        assert actual["experiment"]["name"] == run["label"]
        assert actual["train"]["output_dir"] == str(Path(run["experiment_directory"]).parent)
        checked_hashes = {}
        for path_key, hash_key in (("native_source", "native_source_sha256"),
                                  ("config_source", "config_source_sha256"),
                                  ("input_config", "input_config_sha256"),
                                  ("baseline_config", "baseline_config_sha256"),
                                  ("wrapper", "wrapper_sha256")):
            current = sha(run[path_key])
            assert current == run[hash_key], run[path_key]
            checked_hashes[run[path_key]] = current
        shared = json.loads(Path(run["config_source"]).read_text())
        local = json.loads(Path(run["input_config"]).read_text())
        expected_local = json.loads(json.dumps(shared))
        expected_local["train"]["seed"] = seed
        compare(local, expected_local)
        row = {"training_seed": seed, "native_effective_config": str(native_configs[0]),
               "native_effective_config_sha256": sha(native_configs[0]),
               "configuration_matches_prior_baseline": True,
               "only_local_input_seed_changed": True, "shared_configuration_unchanged": True,
               "source_hashes_unchanged": checked_hashes, "epochs": 2000,
               "updates_per_epoch": 100, "batch_size": 100,
               "log": read_epochs(folder / "training.log", args.complete)}
        if args.complete:
            import torch
            torch.set_num_threads(1)
            summary = json.loads((folder / "summary.json").read_text())
            assert summary["status"] == "completed" and summary["seed"] == seed
            assert summary["configuration_matches_baseline"] is True
            checkpoint = Path(summary["checkpoint"])
            assert checkpoint.name == "model_epoch_2000.pth"
            assert checkpoint.parent.parent / "config.json" == native_configs[0]
            assert sha(checkpoint) == summary["checkpoint_sha256"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert payload["algo_name"] == "bc"
            embedded = json.loads(payload["config"]) if isinstance(payload["config"], str) else payload["config"]
            compare(embedded, actual)
            tensors = payload["model"]
            assert tensors and all(torch.is_tensor(tensor) for tensor in tensors.values())
            assert all(tensor.device.type == "cpu" and bool(torch.isfinite(tensor).all()) for tensor in tensors.values())
            assert payload["shape_metadata"]["ac_dim"] == 7
            assert dict(payload["shape_metadata"]["all_shapes"]) == EXPECTED_SHAPES
            assert payload["env_metadata"]["env_name"] == run["task"]
            published = ROOT / "models" / f"human100_mimic1000_seed{seed}.pth"
            if published.exists():
                assert sha(published) == summary["checkpoint_sha256"]
            row["checkpoint"] = {"path": str(checkpoint), "sha256": summary["checkpoint_sha256"],
                                 "epoch": 2000, "embedded_seed": embedded["train"]["seed"],
                                 "embedded_config_matches_actual": True,
                                 "tensor_count": len(tensors),
                                 "tensor_elements": sum(tensor.numel() for tensor in tensors.values()),
                                 "all_weights_finite_on_cpu": True, "action_dimension": 7,
                                 "observation_dimensions": EXPECTED_SHAPES,
                                 "local_model_copy_hash_checked": published.exists()}
        result["runs"].append(row)
    # An atomic replace prevents consumers from reading a partial report.
    target = ROOT / ("training_audit.json" if args.complete else "training_audit_partial.json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    lines = ["학습 기록·설정 감사", "", "검사 범위: " + ("두 학습 완료 및 최종 모델" if args.complete else "진행 중 로그 스냅샷"),
             "사람 100개와 Mimic 성공 1,000개로 구성된 동일한 1,100개 파일 및 SHA256을 확인했다.",
             "두 seed의 실제 native config는 기존 조건과 데이터·출력 경로·이름을 제외하고 일치한다.",
             "2,000 epoch, epoch당 100 update, batch 100이다. 공유 설정·원본 코드·로컬 설정은 시작 해시와 같다."]
    for row in result["runs"]:
        log = row["log"]
        lines.append(f"seed {row['training_seed']}: epoch 1부터 {log['completed_logged_epochs']}까지의 기록이 누락·중복 없이 연속이며 수치 지표가 모두 유한하다. 최근 100 epoch 평균 loss={log['latest_100_loss_mean']:.6f}.")
        if args.complete:
            ckpt = row["checkpoint"]
            lines.append(f"  최종 epoch 2000의 내장 seed·구성·SHA256을 대조했고 {ckpt['tensor_count']}개 텐서({ckpt['tensor_elements']:,}개 원소)는 CPU에서 모두 유한하다.")
    lines.extend(["", "Loss는 GMM 음의 로그우도이므로 음수가 가능하다. 학습 loss만으로 성공률을 판단하지 않는다.",
                  "훈련·GPU 실행·설정 변경·Git 변경 없이 읽기 검사했으며 감사 보고서만 생성했다."])
    target.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "passed", "report": str(target),
                      "logged_epochs": {row["training_seed"]: row["log"]["completed_logged_epochs"] for row in result["runs"]}}, indent=2))


if __name__ == "__main__":
    main()
