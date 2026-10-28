#!/usr/bin/env python3
"""Combine original human demonstrations with a fixed successful Mimic subset.

Inputs are read-only. The output and manifest must not exist. This utility only
copies records; native annotation, generation, training, and evaluation are
unchanged. Example:
    python prepare_mimic_training.py --human human.hdf5 --mimic mimic.hdf5 \
        --output human_mimic500.hdf5 --manifest human_mimic500.json
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np


TASK = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"
MIMIC_TASK = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0"
OBS_DIMS = {
    "eef_pos": 3,
    "eef_quat": 4,
    "gripper_pos": 2,
    "toy_truck_positions": 3,
    "toy_truck_orientations": 4,
    "box_positions": 3,
    "box_orientations": 4,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def demo_names(data):
    names = list(data)
    if any(re.fullmatch(r"demo_\d+", name) is None for name in names):
        raise ValueError(f"Unexpected episode name in {data.file.filename}")
    return sorted(names, key=lambda name: int(name[5:]))


def is_success(demo):
    value = np.asarray(demo.attrs.get("success", None))
    if value.shape != () or value.dtype.kind not in "biuf" or value.item() not in (0, 1):
        raise ValueError(f"{demo.name}: missing or invalid success label")
    return bool(value.item())


def validate_demo(demo):
    if not is_success(demo):
        raise ValueError(f"{demo.name}: unsuccessful selected demonstration")
    actions = demo["actions"]
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] == 0:
        raise ValueError(f"{demo.name}: expected nonempty actions with shape (N, 7)")
    steps = actions.shape[0]
    samples = np.asarray(demo.attrs.get("num_samples", None))
    if samples.shape != () or samples.dtype.kind not in "iu" or int(samples) != steps:
        raise ValueError(f"{demo.name}: num_samples does not match actions")
    arrays = {"actions": actions}
    for key, dim in OBS_DIMS.items():
        obs = demo[f"obs/{key}"]
        if obs.shape != (steps, dim):
            raise ValueError(f"{demo.name}/obs/{key}: expected {(steps, dim)}, got {obs.shape}")
        arrays[f"obs/{key}"] = obs
    for key, array in arrays.items():
        if array.dtype.kind not in "biuf" or not np.isfinite(array[...]).all():
            raise ValueError(f"{demo.name}/{key}: nonnumeric or nonfinite values")
    return steps


def prepare(args):
    human, mimic, output, manifest = (
        Path(getattr(args, key)).expanduser().resolve()
        for key in ("human", "mimic", "output", "manifest")
    )
    if not 1 <= args.limit <= 1000:
        raise ValueError("--limit must be between 1 and 1000")
    if len({human, mimic, output, manifest}) != 4:
        raise ValueError("Input, output, and manifest paths must all differ")
    for path in (output, manifest):
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    hashes = {"human": sha256(human), "mimic": sha256(mimic)}

    with h5py.File(human, "r") as human_file, h5py.File(mimic, "r") as mimic_file:
        human_data, mimic_data = human_file["data"], mimic_file["data"]
        if json.loads(human_data.attrs["env_args"])["env_name"] != TASK:
            raise ValueError("Human dataset must use the original Toy2Box task")
        if json.loads(mimic_data.attrs["env_args"])["env_name"] != MIMIC_TASK:
            raise ValueError("Mimic dataset must use the Toy2Box Mimic task")
        human_names = demo_names(human_data)
        if not 20 <= len(human_names) <= 100:
            raise ValueError(f"Expected 20-100 human demonstrations, got {len(human_names)}")
        mimic_names = demo_names(mimic_data)
        successful = [name for name in mimic_names if is_success(mimic_data[name])]
        if len(successful) < args.limit:
            raise ValueError(f"Need {args.limit} successful Mimic demos; only {len(successful)} available")
        selections = [("human", human_data, human_names), ("mimic", mimic_data, successful[:args.limit])]
        provenance = []
        for source, data, names in selections:
            for name in names:
                provenance.append({
                    "output_demo": f"demo_{len(provenance)}",
                    "source": source,
                    "source_file": str(human if source == "human" else mimic),
                    "source_demo": name,
                    "num_samples": validate_demo(data[name]),
                })
        total = sum(item["num_samples"] for item in provenance)

        with tempfile.TemporaryDirectory(prefix=".prepare_mimic_", dir=output.parent) as data_tmp, \
                tempfile.TemporaryDirectory(prefix=".prepare_mimic_", dir=manifest.parent) as manifest_tmp:
            temporary_data = Path(data_tmp) / "dataset.hdf5"
            temporary_manifest = Path(manifest_tmp) / "manifest.json"
            with h5py.File(temporary_data, "x") as combined:
                for key, value in human_file.attrs.items():
                    combined.attrs[key] = value
                data = combined.create_group("data")
                for key, value in human_data.attrs.items():
                    data.attrs[key] = value
                data.attrs["total"] = total
                sources = {"human": human_data, "mimic": mimic_data}
                for item in provenance:
                    sources[item["source"]].copy(item["source_demo"], data, name=item["output_demo"])
            if hashes != {"human": sha256(human), "mimic": sha256(mimic)}:
                raise RuntimeError("An input changed during preparation; no output published")
            result = {
                "schema": "toy2box.mimic_training_dataset.v1",
                "selection": "All human demos, then the first successful Mimic demos in numeric demo order",
                "human_provenance": "Direct keyboard demonstrations collected under a consistent task-stage protocol; this utility validates stored arrays only",
                "limit": args.limit,
                "sources": {
                    "human": {"path": str(human), "sha256": hashes["human"], "demo_count": len(human_names)},
                    "mimic": {"path": str(mimic), "sha256": hashes["mimic"],
                              "demo_count": len(mimic_names), "successful_count": len(successful)},
                },
                "output": {"path": str(output), "sha256": sha256(temporary_data),
                           "human_count": len(human_names), "mimic_count": args.limit,
                           "demo_count": len(provenance), "total_steps": total, "env_name": TASK},
                "validation": {"all_selected_successful": True, "actions_dim": 7,
                               "observation_dims": OBS_DIMS, "lengths_and_finite_values_checked": True},
                "episodes": provenance,
            }
            with temporary_manifest.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            # Hard-link publication is atomic and exclusive. If the second link
            # fails, remove only the dataset link created by this invocation.
            published_data = False
            try:
                os.link(temporary_data, output)
                published_data = True
                os.link(temporary_manifest, manifest)
            except BaseException:
                if published_data and output.exists() and os.path.samefile(output, temporary_data):
                    output.unlink()
                raise
    return result["output"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("human", "mimic", "output", "manifest"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    try:
        result = prepare(args)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
