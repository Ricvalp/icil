from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


_DEFAULT_WORKSPACE_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.5))
_PROPRIO_KEYS = ("gripper_pose", "gripper_open")
_ACTION_KEYS = ("gripper_pose", "gripper_open")
_MASK_NAMES_TO_IGNORE = (
    "Floor",
    "Wall1",
    "Wall2",
    "Wall3",
    "Wall4",
    "Roof",
    "workspace",
    "diningTable_visible",
)
_MASK_NAME_SUBSTRINGS_TO_IGNORE = (
    "floor",
    "wall",
    "roof",
    "workspace",
    "table",
    "panda_link",
)


def _parse_index(path: Path, prefix: str) -> int:
    return int(path.name.replace(prefix, ""))


def _task_dirs(raw_root: Path, tasks: Sequence[str]) -> List[Path]:
    if tasks:
        out = [raw_root / task for task in tasks]
    else:
        out = sorted(p for p in raw_root.iterdir() if p.is_dir())
    missing = [str(p) for p in out if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing raw task directories: {missing}")
    return out


def _variation_dirs(task_dir: Path, start: int, count: int) -> List[Path]:
    dirs = sorted(
        (p for p in task_dir.glob("variation*") if p.is_dir()),
        key=lambda p: _parse_index(p, "variation"),
    )
    dirs = [p for p in dirs if _parse_index(p, "variation") >= int(start)]
    if int(count) >= 0:
        dirs = dirs[: int(count)]
    return dirs


def _episode_dirs(variation_dir: Path) -> List[Path]:
    episodes_dir = variation_dir / "episodes"
    if not episodes_dir.is_dir():
        return []
    return sorted(
        (p for p in episodes_dir.glob("episode*") if p.is_dir()),
        key=lambda p: _parse_index(p, "episode"),
    )


def _load_low_dim(episode_dir: Path) -> Sequence[Any]:
    path = episode_dir / "low_dim_obs.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing low_dim_obs.pkl: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _vector(obs: Any) -> np.ndarray:
    pose = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(-1)
    gripper_open = np.asarray([obs.gripper_open], dtype=np.float32)
    return np.concatenate([pose, gripper_open], axis=0).astype(np.float32)


def _load_label_map(variation_dir: Path) -> Dict[int, str]:
    path = variation_dir / "mask_to_label.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def _ignore_ids(label_map: Dict[int, str]) -> Tuple[int, ...]:
    out = set()
    for handle, name in label_map.items():
        if name in _MASK_NAMES_TO_IGNORE:
            out.add(int(handle))
            continue
        lower = str(name).lower()
        if any(token in lower for token in _MASK_NAME_SUBSTRINGS_TO_IGNORE):
            out.add(int(handle))
    return tuple(sorted(out))


def _filter_bounds(points: np.ndarray, bounds: Optional[Sequence[Sequence[float]]]) -> np.ndarray:
    if bounds is None or points.shape[0] == 0:
        return np.ones((points.shape[0],), dtype=np.bool_)
    xb, yb, zb = bounds
    return (
        (points[:, 0] >= float(xb[0]))
        & (points[:, 0] <= float(xb[1]))
        & (points[:, 1] >= float(yb[0]))
        & (points[:, 1] <= float(yb[1]))
        & (points[:, 2] >= float(zb[0]))
        & (points[:, 2] <= float(zb[1]))
    )


def _filter_ignore_ids(masks: np.ndarray, ignore_ids: Sequence[int]) -> np.ndarray:
    keep = np.ones((masks.shape[0],), dtype=np.bool_)
    for mask_id in ignore_ids:
        keep &= masks != int(mask_id)
    return keep


def _sample_fixed(
    rng: np.random.Generator,
    points: np.ndarray,
    colors: np.ndarray,
    masks: np.ndarray,
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return (
            np.zeros((num_points, 3), dtype=np.float32),
            np.zeros((num_points, 3), dtype=np.uint8),
            np.zeros((num_points,), dtype=np.int32),
            np.zeros((num_points,), dtype=np.bool_),
        )
    replace = points.shape[0] < int(num_points)
    idx = rng.choice(points.shape[0], size=int(num_points), replace=replace).astype(np.int64)
    valid = np.ones((int(num_points),), dtype=np.bool_)
    return (
        points[idx].astype(np.float32, copy=False),
        colors[idx].astype(np.uint8, copy=False),
        masks[idx].astype(np.int32, copy=False),
        valid,
    )


def _load_point_frame(
    path: Path,
    *,
    rng: np.random.Generator,
    num_points: int,
    ignore_ids: Sequence[int],
    workspace_bounds: Optional[Sequence[Sequence[float]]],
) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing merged point cloud frame: {path}")
    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(data["colors"], dtype=np.uint8).reshape(-1, 3)
        masks = np.asarray(data["masks"], dtype=np.int32).reshape(-1)
    finite = np.isfinite(points).all(axis=1)
    points, colors, masks = points[finite], colors[finite], masks[finite]
    keep = _filter_ignore_ids(masks, ignore_ids)
    points, colors, masks = points[keep], colors[keep], masks[keep]
    keep = _filter_bounds(points, workspace_bounds)
    points, colors, masks = points[keep], colors[keep], masks[keep]
    xyz, rgb, mask_id, valid = _sample_fixed(rng, points, colors, masks, int(num_points))
    return {
        "xyz": xyz,
        "rgb": rgb,
        "mask_id": mask_id,
        "valid": valid,
    }


def _episode_to_arrays(
    episode_dir: Path,
    *,
    rng: np.random.Generator,
    num_points: int,
    ignore_ids: Sequence[int],
    workspace_bounds: Optional[Sequence[Sequence[float]]],
    include_rgb: bool,
    include_mask_id: bool,
) -> Dict[str, np.ndarray]:
    low_dim = _load_low_dim(episode_dir)
    pc_files = sorted((episode_dir / "merged_point_cloud").glob("*.npz"), key=lambda p: int(p.stem))
    T = min(len(pc_files), len(low_dim))
    if T <= 0:
        raise RuntimeError(f"Empty episode: {episode_dir}")
    frames: List[Dict[str, np.ndarray]] = []
    for t in range(T):
        obs = low_dim[t]
        frame = _load_point_frame(
            pc_files[t],
            rng=rng,
            num_points=int(num_points),
            ignore_ids=ignore_ids,
            workspace_bounds=workspace_bounds,
        )
        vec = _vector(obs)
        frame["state"] = vec
        frame["action"] = vec
        if not include_rgb:
            frame.pop("rgb", None)
        if not include_mask_id:
            frame.pop("mask_id", None)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No frames found in episode: {episode_dir}")
    return {key: np.stack([frame[key] for frame in frames], axis=0) for key in frames[0].keys()}


def _dataset_kwargs(compression: Optional[str]) -> Dict[str, Any]:
    if compression is None or compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4}
    return {"compression": str(compression)}


def _create_dataset(group: h5py.Group, name: str, arr: np.ndarray, compression: Optional[str]) -> None:
    kwargs = _dataset_kwargs(compression)
    if name == "xyz":
        data = arr.astype(np.float16, copy=False)
        group.create_dataset(name, data=data, dtype="f2", chunks=(1, data.shape[1], 3), **kwargs)
    elif name == "valid":
        data = arr.astype(np.bool_, copy=False)
        group.create_dataset(name, data=data, dtype="?", chunks=(1, data.shape[1]), **kwargs)
    elif name == "rgb":
        data = arr.astype(np.uint8, copy=False)
        group.create_dataset(name, data=data, dtype="u1", chunks=(1, data.shape[1], 3), **kwargs)
    elif name == "mask_id":
        data = arr.astype(np.int32, copy=False)
        group.create_dataset(name, data=data, dtype="i4", chunks=(1, data.shape[1]), **kwargs)
    elif name in ("state", "action"):
        data = arr.astype(np.float32, copy=False)
        chunk_t = max(1, min(int(data.shape[0]), 64))
        group.create_dataset(name, data=data, dtype="f4", chunks=(chunk_t, data.shape[1]), **kwargs)
    else:
        group.create_dataset(name, data=arr, **kwargs)


def _write_variation(
    output_path: Path,
    *,
    task_name: str,
    variation: int,
    raw_variation_dir: Path,
    episodes: Sequence[Dict[str, np.ndarray]],
    num_points: int,
    ignore_ids: Sequence[int],
    compression: Optional[str],
) -> None:
    tmp_path = output_path.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as h:
        h.attrs["task"] = str(task_name)
        h.attrs["variation"] = int(variation)
        h.attrs["N"] = int(num_points)
        h.attrs["proprio_keys"] = json.dumps(list(_PROPRIO_KEYS))
        h.attrs["action_keys"] = json.dumps(list(_ACTION_KEYS))
        h.attrs["ignore_ids"] = np.asarray(list(ignore_ids), dtype=np.int64)
        episode_ids = np.arange(len(episodes), dtype=np.int64)
        h.create_dataset("episode_ids", data=episode_ids, maxshape=(None,), dtype="i8")
        root = h.create_group("episodes")
        for episode_id, episode in enumerate(episodes):
            group = root.create_group(str(int(episode_id)))
            group.attrs["episode_id"] = int(episode_id)
            group.attrs["T"] = int(episode["action"].shape[0])
            group.attrs["N"] = int(num_points)
            for name, arr in episode.items():
                _create_dataset(group, name, arr, compression)
    tmp_path.replace(output_path)


def _workspace_bounds(args: argparse.Namespace) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    if args.no_workspace_filter:
        return None
    return _DEFAULT_WORKSPACE_BOUNDS


def convert(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    cache_root = Path(args.cache_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else str(args.compression)
    workspace_bounds = _workspace_bounds(args)
    for task_dir in _task_dirs(raw_root, tuple(args.tasks)):
        task_name = task_dir.name
        out_task_dir = cache_root / task_name
        out_task_dir.mkdir(parents=True, exist_ok=True)
        variations = _variation_dirs(task_dir, int(args.start_variation), int(args.variations))
        print(f"Converting task={task_name} variations={len(variations)} raw={task_dir}", flush=True)
        for variation_dir in variations:
            variation = _parse_index(variation_dir, "variation")
            out_path = out_task_dir / f"variation{variation}.h5"
            if out_path.exists() and not args.overwrite:
                print(f"skip existing {out_path}", flush=True)
                continue
            label_map = _load_label_map(variation_dir)
            ignore_ids = _ignore_ids(label_map)
            episode_dirs = _episode_dirs(variation_dir)
            if not episode_dirs:
                raise RuntimeError(f"No episodes found in {variation_dir}")
            episodes: List[Dict[str, np.ndarray]] = []
            print(
                f"  variation {variation}: episodes={len(episode_dirs)} ignore_ids={list(ignore_ids)}",
                flush=True,
            )
            for episode_dir in episode_dirs:
                episode_id = _parse_index(episode_dir, "episode")
                seed = int(args.seed) + episode_id
                rng = np.random.default_rng(seed)
                episode = _episode_to_arrays(
                    episode_dir,
                    rng=rng,
                    num_points=int(args.num_points),
                    ignore_ids=ignore_ids,
                    workspace_bounds=workspace_bounds,
                    include_rgb=bool(args.include_rgb),
                    include_mask_id=bool(args.include_mask_id),
                )
                episodes.append(episode)
            _write_variation(
                out_path,
                task_name=task_name,
                variation=variation,
                raw_variation_dir=variation_dir,
                episodes=episodes,
                num_points=int(args.num_points),
                ignore_ids=ignore_ids,
                compression=compression,
            )
            lengths = [int(ep["action"].shape[0]) for ep in episodes]
            print(f"  wrote {out_path} T=[{min(lengths)}, {max(lengths)}]", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw RLBench dataset_generator_pc episodes to dense H5 cache.")
    parser.add_argument("--raw-root", required=True, help="Raw RLBench dataset root produced by rlbench.dataset_generator_pc.")
    parser.add_argument("--cache-root", required=True, help="Output dense H5 cache root.")
    parser.add_argument("--tasks", nargs="*", default=(), help="Task directories to convert. Defaults to all tasks under raw-root.")
    parser.add_argument("--start-variation", type=int, default=0, help="First variation index to convert.")
    parser.add_argument("--variations", type=int, default=-1, help="Number of variations to convert. -1 converts all from start-variation.")
    parser.add_argument("--num-points", type=int, default=1024, help="Fixed points per frame in the dense cache.")
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-rgb", dest="include_rgb", action="store_true", default=True)
    parser.add_argument("--no-rgb", dest="include_rgb", action="store_false")
    parser.add_argument("--include-mask-id", dest="include_mask_id", action="store_true", default=True)
    parser.add_argument("--no-mask-id", dest="include_mask_id", action="store_false")
    parser.add_argument("--no-workspace-filter", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
