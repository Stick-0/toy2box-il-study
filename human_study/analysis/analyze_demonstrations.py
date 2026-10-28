#!/usr/bin/env python3
"""Post-hoc, read-only analysis of existing human30 and successful Mimic500 data.

No simulator, new rollout, model training, or changes to the source HDF5 files.
Run from any directory; requires numpy, h5py and matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/toy2box-data-analysis-matplotlib")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "human_study/data"
SOURCES = {
    "human30": DATA / "human_20261028_112300.hdf5",
    "mimic500": DATA / "human30_mimic500_raw_20261028_env8.hdf5",
}
# Configured world-frame reset range, transformed into the fixed robot base
# frame (robot translation = [-0.6, 0, -1.05], identity quaternion).
TOY_LOW = np.array([0.45, -0.30, -3.14])
TOY_HIGH = np.array([0.80, -0.15, 3.14])
GRID_BINS = np.array([4, 3, 8])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                min=float(x.min()), max=float(x.max()),
                p25=float(np.percentile(x, 25)), p75=float(np.percentile(x, 75)),
                sd_sample=float(np.std(x, ddof=1)) if len(x) > 1 else None)


def yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z)))


def bin_ids(poses, bins=GRID_BINS):
    bins = np.asarray(bins)
    idx = np.floor((poses - TOY_LOW) / (TOY_HIGH - TOY_LOW) * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    return np.ravel_multi_index(idx.T, bins)


def load_dataset(label, path):
    rows = []
    with h5py.File(path, "r") as h:
        args = json.loads(h["data"].attrs["env_args"])
        timestep = args["sim_args"]["dt"] * args["sim_args"]["decimation"]
        for name in sorted(h["data"], key=lambda k: int(k.split("_")[-1])):
            d = h["data"][name]
            a = d["actions"][:].astype(float)
            ee = d["obs/eef_pos"][:].astype(float)
            toy = d["obs/toy_truck_positions"][:].astype(float)
            box = d["obs/box_positions"][:].astype(float)
            assert np.isfinite(a).all() and np.isfinite(ee).all()
            assert bool(d.attrs["success"])
            closed = a[:, 6] < 0
            transition = np.diff(closed.astype(int))
            closes = np.flatnonzero(transition == 1) + 1
            opens = np.flatnonzero(transition == -1) + 1
            firstclose = int(closes[0]) if len(closes) else len(a) - 1
            close_rel = ee[firstclose] - toy[firstclose]
            zero = np.linalg.norm(a[:, :6], axis=1) < 1e-6
            firstactive = int(np.flatnonzero(~zero)[0]) if (~zero).any() else len(a)
            pre = ee[:firstclose+1]
            pre_steps = np.diff(pre, axis=0)
            path = np.linalg.norm(pre_steps, axis=1).sum()
            chord = np.linalg.norm(pre[-1] - pre[0])
            # Corrections along each axis, ignoring near-zero commands. This is
            # descriptive: sequential approach/align motions can legitimately
            # reverse an axis; counts are NOT a demonstrated jitter detector.
            flips = []
            for axis in range(3):
                active = a[:firstclose, axis]
                active = active[np.abs(active) > 1e-5]
                flips.append(int(np.sum(active[1:] * active[:-1] < 0)))
            row = dict(dataset=label, demo=name, steps=len(a), duration_s=len(a)*timestep,
                       first_close_step=firstclose, first_close_fraction=firstclose/len(a),
                       close_events=len(closes), open_events=len(opens),
                       starts_open=not bool(closed[0]), ends_open=not bool(closed[-1]),
                       arm_zero_steps=int(zero.sum()), arm_zero_fraction=float(zero.mean()),
                       leading_arm_zero_steps=firstactive,
                       preclose_arm_zero_fraction=float(zero[:firstclose].mean()),
                       preclose_ee_path_m=float(path), preclose_ee_chord_m=float(chord),
                       preclose_path_chord_ratio=float(path/chord),
                       preclose_x_command_sign_flips=flips[0],
                       preclose_y_command_sign_flips=flips[1],
                       preclose_z_command_sign_flips=flips[2],
                       close_ee_toy_xy_m=float(np.linalg.norm(close_rel[:2])),
                       close_ee_toy_z_m=float(close_rel[2]),
                       toy_initial_x_m=float(toy[0, 0]), toy_initial_y_m=float(toy[0, 1]),
                       toy_initial_yaw_rad=yaw(d["obs/toy_truck_orientations"][0]),
                       box_initial_x_m=float(box[0, 0]), box_initial_y_m=float(box[0, 1]),
                       ee_initial_x_m=float(ee[0, 0]), ee_initial_y_m=float(ee[0, 1]),
                       ee_initial_z_m=float(ee[0, 2]))
            rows.append(row)
        assert sum(r["steps"] for r in rows) == int(h["data"].attrs["total"])
    return rows


def get_poses(rows):
    return np.array([[r["toy_initial_x_m"], r["toy_initial_y_m"], r["toy_initial_yaw_rad"]] for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = {k: load_dataset(k, v) for k, v in SOURCES.items()}
    poses = {k: get_poses(v) for k, v in rows.items()}
    poses["combined530"] = np.concatenate(list(poses.values()))
    with open(out / "demonstration_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows["human30"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows["human30"] + rows["mimic500"])

    # Fixed, deterministic geometric probe points. These are NOT simulations,
    # evaluation observations, rollout seeds, or an estimate of success rate.
    dims = (20, 12, 24)
    axes = [lo + (np.arange(n) + 0.5) / n * (hi-lo)
            for lo, hi, n in zip(TOY_LOW, TOY_HIGH, dims)]
    probes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    coverage = {}
    sensitivity = {"occupancy": [], "geometric_proximity": []}
    for bins in [(2, 2, 4), (4, 3, 8), (6, 4, 12)]:
        sensitivity["occupancy"].append({
            "bins": bins, "total_bins": int(np.prod(bins)),
            **{k: int(len(np.unique(bin_ids(v, bins)))) for k, v in poses.items()}})
    near_by_source = {}
    grid_distances = {}
    for name, p in poses.items():
        xy = np.linalg.norm(probes[:, None, :2] - p[None, :, :2], axis=-1)
        angle = np.abs(probes[:, None, 2] - p[None, :, 2])
        angle = np.minimum(angle, 2*np.pi-angle)
        near = (xy <= 0.03) & (angle <= np.deg2rad(30))
        near_by_source[name] = near
        for xy_m, yaw_deg in [(0.02, 15), (0.03, 30), (0.05, 45)]:
            sensitivity["geometric_proximity"].append({
                "dataset": name, "xy_threshold_m": xy_m, "yaw_threshold_deg": yaw_deg,
                "probe_near_fraction": float(((xy <= xy_m) & (angle <= np.deg2rad(yaw_deg))).any(axis=1).mean())})
        nearest = xy.min(axis=1)
        grid_distances[name] = nearest
        bins = bin_ids(p)
        coverage[name] = dict(
            n=len(p), occupied_toy_xy_yaw_bins=int(len(np.unique(bins))),
            total_toy_xy_yaw_bins=int(np.prod(GRID_BINS)),
            occupied_toy_xy_bins=int(len(np.unique(bins // GRID_BINS[2]))),
            occupied_toy_yaw_bins=int(len(np.unique(bins % GRID_BINS[2]))),
            pose_min=p.min(axis=0).tolist(), pose_max=p.max(axis=0).tolist(),
            probe_nearest_xy_m=summary(nearest),
            probe_fraction_near_xy_and_yaw=float(near.any(axis=1).mean()),
            probe_fraction_near_xy=float((xy <= 0.03).any(axis=1).mean()),
        )
    # Equal N removes the automatic benefit of having 500 rather than 30
    # observations. It cannot remove successful-generation selection bias or
    # dependence between source-derived demonstrations.
    rng = np.random.default_rng(20261029)
    sub_occupancy, sub_near = [], []
    sub_rows = []
    for repeat in range(1000):
        ids = rng.choice(500, size=30, replace=False)
        occ = len(np.unique(bin_ids(poses["mimic500"][ids])))
        near = near_by_source["mimic500"][:, ids].any(axis=1).mean()
        sub_occupancy.append(occ)
        sub_near.append(float(near))
        sub_rows.append(dict(repeat=repeat, n=30, occupied_bins=occ, probe_near_fraction=float(near)))
    with open(out / "matched_size_subsamples.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sub_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sub_rows)

    metrics = ["steps", "duration_s", "arm_zero_fraction", "leading_arm_zero_steps",
               "preclose_arm_zero_fraction", "first_close_fraction", "preclose_path_chord_ratio",
               "close_ee_toy_xy_m", "close_ee_toy_z_m", "preclose_x_command_sign_flips",
               "preclose_y_command_sign_flips", "preclose_z_command_sign_flips"]
    dataset_stats = {}
    for name, rr in rows.items():
        dataset_stats[name] = {
            "demonstrations": len(rr), "steps": sum(r["steps"] for r in rr),
            "single_close_then_open_demonstrations": sum(
                r["starts_open"] and r["ends_open"] and r["close_events"] == 1 and r["open_events"] == 1 for r in rr),
            "arm_zero_fraction_step_weighted": sum(r["arm_zero_steps"] for r in rr) / sum(r["steps"] for r in rr),
            "metrics": {m: summary([r[m] for r in rr]) for m in metrics},
            "initial_ee_xyz_spread_m": (np.ptp([[r["ee_initial_x_m"], r["ee_initial_y_m"], r["ee_initial_z_m"]] for r in rr], axis=0)).tolist(),
        }
    combined_fraction = len(rows["human30"])/(len(rows["human30"])+len(rows["mimic500"]))
    result = dict(
        schema="toy2box.posthoc_demonstration_analysis.v1", analysis_date="2026-10-29",
        analysis_type="Retrospective descriptive analysis; no additional training or rollouts",
        sources={k: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for k,p in SOURCES.items()},
        action_semantics="3D relative Cartesian displacement (m), 3D axis-angle displacement (rad), binary gripper (-1 close, +1 open); scale 1; 20 Hz",
        observation_semantics="Pre-step EE and toy poses in fixed robot-base coordinates; action is desired target delta, not observed motion",
        occupancy={"toy_range_base_frame": [TOY_LOW.tolist(), TOY_HIGH.tolist()], "bins": GRID_BINS.tolist(), "bin_widths": ((TOY_HIGH-TOY_LOW)/GRID_BINS).tolist()},
        geometric_probes={"dimensions": dims, "n": len(probes), "xy_threshold_m": 0.03, "yaw_threshold_deg": 30,
                          "meaning": "Deterministic uniformly spaced points in configured initial toy pose domain; not rollout observations or success estimates"},
        datasets=dataset_stats, coverage=coverage, sensitivity=sensitivity,
        matched_size_mimic30={"resamples": 1000, "seed": 20261029,
                              "occupied_bins": summary(sub_occupancy), "probe_near_fraction": summary(sub_near),
                              "occupied_bins_2p5_97p5_percentiles": np.percentile(sub_occupancy, [2.5, 97.5]).tolist(),
                              "occupancy_percentile_of_human30": float(np.mean(np.asarray(sub_occupancy) <= coverage["human30"]["occupied_toy_xy_yaw_bins"])),
                              "interpretation": "Descriptive equal-size subsampling; resamples overlap and are not independent experiments or a statistical test"},
        training_mix={"human_demo_fraction": combined_fraction,
                      "human_step_fraction": dataset_stats["human30"]["steps"] / sum(d["steps"] for d in dataset_stats.values())},
        limitations=[
            "Initial pose coverage does not measure evaluation-time policy state coverage or prove the cause of failures.",
            "The fixed initial EE pose makes initial EE-to-toy XY coverage equivalent to initial toy XY coverage, not independent evidence.",
            "Coarse bins and geometric proximity thresholds are analyst choices; occupancy depends on sample size.",
            "Only successful Mimic demonstrations used in training are analyzed; generation selection bias and reuse of 28 annotated source demonstrations remain.",
            "A single close/open sequence supports coarse action-order consistency, not correct grasp alignment or expert optimality.",
            "Zero arm action may be a deliberate hold; a path/chord ratio above 1 includes intended alignment and descent; axis sign changes are not automatically jitter.",
            "Mimic retargeting, interpolation and noise change action distribution; raw action smoothness is not directly comparable to discrete keyboard actions.",
            "The intervention combines extra samples, pose coverage, changed action targets and changed human/synthetic mixture; it does not isolate coverage from sample count or noise.",
            "No evaluation state trajectories are available to link individual rollout failures to distance from the training pose distribution."
        ])
    (out / "demonstration_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange = "#2166ac", "#d6604d"
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axs[0,0]
    ax.scatter(poses["mimic500"][:,0], poses["mimic500"][:,1], s=12, alpha=.35, color=orange, label="Mimic successes (500)")
    ax.scatter(poses["human30"][:,0], poses["human30"][:,1], s=35, edgecolor="white", linewidth=.5, color=blue, label="Human (30)")
    ax.set(xlabel="Initial toy x in robot base (m)", ylabel="Initial toy y in robot base (m)", title="A  Initial position samples", xlim=(.45,.80), ylim=(-.30,-.15))
    ax.legend(fontsize=9)
    ax = axs[0,1]
    centers = np.arange(8)
    for name, offset, color, label in [("human30",-.18,blue,"Human (30)"),("mimic500",.18,orange,"Mimic (500)")]:
        counts = np.bincount(bin_ids(poses[name])%8, minlength=8)/len(poses[name])*100
        ax.bar(centers+offset, counts, width=.36, color=color, label=label)
    ax.set(xticks=centers, xticklabels=[f"{int(v)}" for v in np.linspace(-157.5,157.5,8)], xlabel="Initial toy yaw bin center (degrees)", ylabel="Demonstrations (%)", title="B  Initial orientation distribution")
    ax.legend(fontsize=9)
    ax = axs[1,0]
    labels = ["Human 30", "Mimic 30\nsubsamples", "Mimic 500", "Combined 530"]
    heights = [coverage["human30"]["occupied_toy_xy_yaw_bins"], np.mean(sub_occupancy), coverage["mimic500"]["occupied_toy_xy_yaw_bins"], coverage["combined530"]["occupied_toy_xy_yaw_bins"]]
    ax.bar(labels, heights, color=[blue,"#f4a582",orange,"#777777"])
    ax.errorbar(1, heights[1], yerr=[[heights[1]-np.percentile(sub_occupancy,2.5)],[np.percentile(sub_occupancy,97.5)-heights[1]]], color="black", capsize=4)
    for i,v in enumerate(heights): ax.text(i,v+3,f"{v:.1f}",ha="center")
    ax.axhline(96,ls="--",color="gray",lw=1)
    ax.set(ylabel="Occupied cells / 96", ylim=(0,106), title="C  Joint x/y/yaw coverage (4 x 3 x 8)")
    ax.text(.03,.96,"Mimic 30: 1,000 subsets; 2.5-97.5% interval",transform=ax.transAxes,fontsize=8,va="top")
    ax = axs[1,1]
    for name,color,label in [("human30",blue,"Human (30)"),("mimic500",orange,"Mimic (500)")]:
        values=np.sort(grid_distances[name])*100
        ax.plot(values,np.arange(1,len(values)+1)/len(values)*100,color=color,label=label,lw=2)
    ax.set(xlabel="Distance to closest sampled initial XY (cm)", ylabel="Geometric probes covered (%)", title="D  Initial-position spacing (fixed domain probes)")
    ax.legend(fontsize=9)
    fig.suptitle("Recorded training data: initial-state coverage\nPost-hoc analysis, 2026-10-29; geometric probes are not evaluation rollouts", fontsize=13)
    fig.savefig(out/"initial_pose_coverage.png",dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for ax, metric, title, xlabel in [
        (axs[0],"arm_zero_fraction","A  Zero arm-command fraction","Fraction of recorded steps"),
        (axs[1],"close_ee_toy_xy_m","B  First close command: relative position","EE-to-toy XY distance (m)"),
        (axs[2],"preclose_path_chord_ratio","C  Actual EE path before close","Path length / endpoint distance")]:
        ax.hist([r[metric] for r in rows["human30"]],bins=8,color=blue,edgecolor="white")
        ax.set(title=title,xlabel=xlabel,ylabel="Human demonstrations")
    fig.suptitle("Human 30: all 30 follow one close then one open command\nHolds and intended alignment motions are not automatically action-quality failures",fontsize=12)
    fig.savefig(out/"human_action_diagnostics.png",dpi=180)
    plt.close(fig)
    print(json.dumps({"datasets":dataset_stats,"coverage":coverage,"matched_size":result["matched_size_mimic30"]},indent=2))


if __name__ == "__main__":
    main()
