#!/usr/bin/env python3
"""Audit existing logs, configs and videos; never launch training or simulation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path: Path):
    return json.loads(path.read_text())


def config_diff(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result.extend(config_diff(a.get(key), b.get(key), f"{prefix}.{key}"))
        return result
    return [] if a == b else [{"field": prefix, "baseline": a, "mimic500": b}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--lab-root", type=Path)
    parser.add_argument("--robomimic-source", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    lab = (args.lab_root or repo.parent / "IsaacLab_RS").resolve()
    run = repo / "human_study/results/mimic500_20261028"
    output = repo / "human_study/analysis"
    output.mkdir(exist_ok=True)
    baseline = json_read(run / "baseline_audit.json")

    def source_ref(path):
        path = path.resolve()
        if path.is_relative_to(repo):
            return {"path": str(path.relative_to(repo)), "scope": "repository", "sha256": sha256(path)}
        if path.is_relative_to(lab):
            return {"path": str(Path("../IsaacLab_RS") / path.relative_to(lab)),
                    "scope": "external_local_IsaacLab_RS", "sha256": sha256(path)}
        return {"path": "site-packages/robomimic/algo/bc.py", "scope": "installed_dependency",
                "sha256": sha256(path)}

    rows, training, configs = [], [], {}
    for condition in ("baseline", "mimic500"):
        for seed in (101, 111):
            if condition == "baseline":
                record = next(item for item in baseline["runs"] if item["training_seed"] == seed)
                old_path = Path(record["training_log_path"])
                relative = old_path.parts[old_path.parts.index("IsaacLab_RS") + 1:]
                log = lab.joinpath(*relative)
                config = run / f"configs/baseline_seed{seed}.json"
            else:
                log = run / f"training/mimic500_seed{seed}/training.log"
                config = run / f"training/mimic500_seed{seed}/effective_config.json"
            matches = re.findall(r"Train Epoch (\d+)\s*(\{.*?\})", log.read_text(), re.S)
            epochs = {int(epoch): json.loads(values) for epoch, values in matches}
            if len(matches) != 2000 or sorted(epochs) != list(range(1, 2001)):
                raise ValueError(f"Expected exactly 2,000 distinct training epochs: {log}")
            losses = [epochs[epoch]["Loss"] for epoch in range(1, 2001)]
            for epoch in range(1, 2001):
                if abs(epochs[epoch]["Loss"] + epochs[epoch]["Log_Likelihood"]) > 1e-8:
                    raise ValueError(f"Loss and log-likelihood disagree: {log}, epoch {epoch}")
                rows.append({"condition": condition, "training_seed": seed, "epoch": epoch,
                             "training_negative_log_likelihood": epochs[epoch]["Loss"]})
            cfg = json_read(config)
            if (cfg["train"]["num_epochs"], cfg["experiment"]["epoch_every_n_steps"],
                    cfg["train"]["batch_size"]) != (2000, 100, 100):
                raise ValueError(f"Unexpected training budget: {config}")
            if not (cfg["algo"]["gmm"]["enabled"] and cfg["algo"]["rnn"]["enabled"]):
                raise ValueError(f"Expected the recorded BC-RNN-GMM policy: {config}")
            configs[(condition, seed)] = cfg
            training.append({"condition": condition, "training_seed": seed, "epochs": len(epochs),
                             "source_log": source_ref(log), "source_config": source_ref(config),
                             "loss_epoch1": losses[0], "loss_epoch2000": losses[-1],
                             "loss_first100_mean": sum(losses[:100]) / 100,
                             "loss_last100_mean": sum(losses[-100:]) / 100,
                             "validate": cfg["experiment"]["validate"],
                             "training_rollouts_enabled": cfg["experiment"]["rollout"]["enabled"],
                             "rnn_open_loop": cfg["algo"]["rnn"]["open_loop"],
                             "rnn_horizon": cfg["algo"]["rnn"]["horizon"]})

    csv_path = output / "training_loss.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    videos = []
    video_runs = []
    for record in baseline["runs"]:
        seed = record["training_seed"]
        folder = lab / "videos" / f"human30_seed{seed}_eval42"
        files = sorted(folder.glob("rollout-episode-*.mp4"),
                       key=lambda path: int(path.stem.rsplit("-", 1)[1]))
        if len(files) != 100:
            raise ValueError(f"Expected 100 existing videos in {folder}")
        for video in files:
            videos.append({"training_seed": seed, "episode": int(video.stem.rsplit("-", 1)[1]),
                           "path": str(video.relative_to(lab)), "bytes": video.stat().st_size,
                           "sha256": sha256(video)})
        original_checkpoint = Path(record["checkpoint_path"])
        checkpoint_parts = original_checkpoint.parts[original_checkpoint.parts.index("IsaacLab_RS") + 1:]
        video_runs.append({"training_seed": seed,
                           "started_at_utc": record["eval_command_provenance"].split()[0],
                           "checkpoint_path": str(Path(*checkpoint_parts)),
                           "evaluation_seed": record["eval_seed"], "rollouts": record["rollouts"],
                           "horizon": record["horizon"], "video_length_limit": record["video_length_limit"],
                           "video_folder": str(folder.relative_to(lab)),
                           "provenance": "baseline_audit.json / runs / eval_command_provenance"})
    counts = Counter(item["sha256"] for item in videos)
    video_audit = {"audit_date": "2026-10-29", "timezone": "Asia/Seoul", "post_hoc": True,
                   "source_root": "IsaacLab_RS (external local project)",
                   "path_basis": "video paths are relative to the source root",
                   "source_audit": source_ref(run / "baseline_audit.json"),
                   "video_count": len(videos), "unique_file_sha256_count": len(counts),
                   "duplicate_sha256": [key for key, count in counts.items() if count > 1],
                   "recorded_runs": video_runs, "videos": videos,
                   "limitations": ["Different encoded-file hashes do not alone prove different visual content.",
                                   "These recording runs are separate from the later logged, non-video evaluations.",
                                   "No video episode is assigned a success label from a different evaluation run."]}
    video_path = output / "video_source_audit.json"
    video_path.write_text(json.dumps(video_audit, ensure_ascii=False, indent=2) + "\n")

    robomimic_source = args.robomimic_source
    if robomimic_source is None:
        spec = importlib.util.find_spec("robomimic")
        if spec is None or spec.origin is None:
            raise RuntimeError("Use the course environment or specify --robomimic-source for the installed bc.py")
        robomimic_source = Path(spec.origin).parent / "algo/bc.py"
    native_play = repo / "IsaacLab/scripts/imitation_learning/robomimic/play.py"
    env_config = repo / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/place_toy2box_rmp_rel_env_cfg.py"
    formal_runs = []
    for condition in ("baseline", "mimic500"):
        for seed in (101, 111):
            folder = run / f"evaluation/{condition}_seed{seed}"
            metadata = json_read(folder / "run.json")
            formal_runs.append({"condition": condition, "training_seed": seed,
                                "started_at_utc": metadata["started_at"], "video": metadata["video"],
                                "record": source_ref(folder / "run.json"),
                                "summary": source_ref(folder / "summary.json"),
                                "trials": source_ref(folder / "trials.jsonl")})
    audit = {"audit_date": "2026-10-29", "timezone": "Asia/Seoul", "post_hoc": True,
             "new_training_or_rollouts": False, "training_runs": training,
             "loss_csv": source_ref(csv_path),
             "loss_definition": "BC_RNN_GMM negative mean log probability of demonstration actions; lower is better on that training distribution.",
             "training_config_differences": {str(seed): config_diff(configs[("baseline", seed)], configs[("mimic500", seed)]) for seed in (101, 111)},
             "equal_training_budget": {"epochs": 2000, "optimizer_updates_per_epoch": 100,
                                       "batch_size": 100, "optimizer_updates_total": 200000},
             "video_audit": source_ref(video_path), "formal_evaluation_runs": formal_runs,
             "code_evidence": {"native_play": source_ref(native_play), "environment_config": source_ref(env_config),
                               "robomimic_bc": source_ref(robomimic_source),
                               "robomimic_symbols": ["BC_RNN.get_action", "BC_RNN_GMM._compute_losses"]},
             "extraction_script": source_ref(Path(__file__)),
             "interpretation": [
                 "Baseline training NLL improved; this does not establish rollout competence or identify the failure cause.",
                 "No validation loss exists because experiment.validate is false; overfitting is not directly demonstrated.",
                 "Training NLL values from different dataset distributions are not a generalization ranking.",
                 "rnn.open_loop=false means current observations are consumed at every step; blind sequence replay is not an established explanation.",
                 "Mimic addition changes dataset size and distribution together; the effect of diversity alone is not isolated.",
                 "Recording runs cannot be joined to later logged evaluation trials solely by seed and episode number."]}
    (output / "evidence_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"training_loss_rows": len(rows), "videos": len(videos),
                      "unique_video_sha256": len(counts), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
