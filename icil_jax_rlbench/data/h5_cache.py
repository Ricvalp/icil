from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class VariationKey:
    task: str
    variation: int
    path: str


def discover_tasks(cache_root: Path) -> List[str]:
    root = Path(cache_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob('variation*.h5')))


def build_variation_keys(cache_root: Path, task: str) -> List[VariationKey]:
    keys: List[VariationKey] = []
    task_dir = Path(cache_root) / str(task)
    for p in sorted(task_dir.glob('variation*.h5')):
        variation = int(p.stem.replace('variation', ''))
        keys.append(VariationKey(task=str(task), variation=variation, path=str(p)))
    return keys


def build_keys(cache_root: Path, tasks: Sequence[str] | None = None, exclude_tasks: Sequence[str] = ()) -> Tuple[List[VariationKey], List[str]]:
    root = Path(cache_root)
    if not root.is_dir():
        raise FileNotFoundError(f'cache_root not found: {root}')
    selected = list(tasks) if tasks else discover_tasks(root)
    exclude = set(str(t) for t in exclude_tasks)
    selected = [t for t in selected if t not in exclude]
    if not selected:
        raise RuntimeError(f'No tasks selected under {root}.')
    keys: List[VariationKey] = []
    missing: List[str] = []
    for task in selected:
        task_keys = build_variation_keys(root, task)
        if not task_keys:
            missing.append(task)
        keys.extend(task_keys)
    if missing:
        raise RuntimeError(f'No variation*.h5 files found for tasks: {missing[:10]}')
    if not keys:
        raise RuntimeError(f'No variation*.h5 files found under {root}.')
    return keys, selected


class RLBenchCacheStore:
    """Standalone RLBench dense H5 cache reader with optional full preload."""

    def __init__(self, keys: Sequence[VariationKey], *, keep_open: bool = True, preload_to_memory: bool = False):
        self.keys = list(keys)
        self.task_names = tuple(sorted({key.task for key in self.keys}))
        self.task_to_id = {task: idx for idx, task in enumerate(self.task_names)}
        self.task_variation_keys = tuple(f'{key.task}:{int(key.variation)}' for key in self.keys)
        self.task_variation_to_id = {key: idx for idx, key in enumerate(self.task_variation_keys)}
        self.keep_open = bool(keep_open) and not bool(preload_to_memory)
        self.preload_to_memory = bool(preload_to_memory)
        self._handles: Dict[int, h5py.File] = {}
        self._preloaded: Dict[int, Dict[str, Any]] = {}
        self.preloaded_bytes = 0
        if self.preload_to_memory:
            self._preload_all()

    def __len__(self) -> int:
        return len(self.keys)

    def close(self) -> None:
        for h in list(self._handles.values()):
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()

    def _handle(self, vidx: int) -> h5py.File:
        vidx = int(vidx)
        if self.keep_open:
            h = self._handles.get(vidx)
            if h is None:
                h = h5py.File(self.keys[vidx].path, 'r')
                self._handles[vidx] = h
            return h
        return h5py.File(self.keys[vidx].path, 'r')

    @staticmethod
    def _read_rows(ds: h5py.Dataset | np.ndarray, t_idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(t_idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return np.asarray(ds[idx])
        if np.all(idx[1:] > idx[:-1]):
            return np.asarray(ds[idx])
        unique_idx, inverse = np.unique(idx, return_inverse=True)
        return np.asarray(ds[unique_idx])[inverse]

    def _preload_one(self, vidx: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {'episodes': {}}
        with h5py.File(self.keys[vidx].path, 'r') as h:
            out['episode_ids'] = np.asarray(h['episode_ids'][:], dtype=np.int64)
            self.preloaded_bytes += int(out['episode_ids'].nbytes)
            for eid in out['episode_ids']:
                g = h['episodes'][str(int(eid))]
                ep: Dict[str, Any] = {'attrs': dict(g.attrs)}
                for name in ('xyz', 'valid', 'state', 'action', 'rgb', 'mask_id'):
                    if name in g:
                        arr = np.asarray(g[name][:])
                        ep[name] = arr
                        self.preloaded_bytes += int(arr.nbytes)
                out['episodes'][int(eid)] = ep
        return out

    def _preload_all(self) -> None:
        for vidx in range(len(self.keys)):
            self._preloaded[vidx] = self._preload_one(vidx)

    def list_episode_ids(self, vidx: int) -> np.ndarray:
        if self.preload_to_memory:
            return np.asarray(self._preloaded[int(vidx)]['episode_ids'], dtype=np.int64)
        if self.keep_open:
            return np.asarray(self._handle(vidx)['episode_ids'][:], dtype=np.int64)
        with self._handle(vidx) as h:
            return np.asarray(h['episode_ids'][:], dtype=np.int64)

    def episode_length(self, vidx: int, episode_id: int) -> int:
        if self.preload_to_memory:
            return int(self._preloaded[int(vidx)]['episodes'][int(episode_id)]['attrs']['T'])
        if self.keep_open:
            return int(self._handle(vidx)['episodes'][str(int(episode_id))].attrs['T'])
        with self._handle(vidx) as h:
            return int(h['episodes'][str(int(episode_id))].attrs['T'])

    def load_episode_slices(
        self,
        vidx: int,
        episode_id: int,
        t_idx: np.ndarray,
        *,
        load_rgb: bool = True,
        load_mask_id: bool = True,
        load_full_traj: bool = False,
    ) -> Dict[str, np.ndarray]:
        vidx = int(vidx)
        eid = int(episode_id)
        idx = np.asarray(t_idx, dtype=np.int64)
        if self.preload_to_memory:
            ep = self._preloaded[vidx]['episodes'][eid]
            out = {
                'xyz': self._read_rows(ep['xyz'], idx).astype(np.float32),
                'valid': self._read_rows(ep['valid'], idx).astype(np.bool_),
                'state': self._read_rows(ep['state'], idx).astype(np.float32),
                'action': self._read_rows(ep['action'], idx).astype(np.float32),
            }
            if load_rgb and 'rgb' in ep:
                out['rgb'] = self._read_rows(ep['rgb'], idx).astype(np.float32) / 255.0
            if load_mask_id and 'mask_id' in ep:
                out['mask_id'] = self._read_rows(ep['mask_id'], idx).astype(np.int32)
            if load_full_traj:
                out['traj'] = np.asarray(ep['action'], dtype=np.float32)
            return out

        def read_from_group(g: h5py.Group) -> Dict[str, np.ndarray]:
            out = {
                'xyz': self._read_rows(g['xyz'], idx).astype(np.float32),
                'valid': self._read_rows(g['valid'], idx).astype(np.bool_),
                'state': self._read_rows(g['state'], idx).astype(np.float32),
                'action': self._read_rows(g['action'], idx).astype(np.float32),
            }
            if load_rgb and 'rgb' in g:
                out['rgb'] = self._read_rows(g['rgb'], idx).astype(np.float32) / 255.0
            if load_mask_id and 'mask_id' in g:
                out['mask_id'] = self._read_rows(g['mask_id'], idx).astype(np.int32)
            if load_full_traj:
                out['traj'] = np.asarray(g['action'][:], dtype=np.float32)
            return out

        if self.keep_open:
            return read_from_group(self._handle(vidx)['episodes'][str(eid)])
        with self._handle(vidx) as h:
            return read_from_group(h['episodes'][str(eid)])

    def infer_dims(self) -> Tuple[int, int, int]:
        for vidx in range(len(self)):
            eids = self.list_episode_ids(vidx)
            if len(eids) == 0:
                continue
            sample = self.load_episode_slices(vidx, int(eids[0]), np.asarray([0]), load_rgb=False, load_mask_id=False)
            return int(sample['xyz'].shape[1]), int(sample['state'].shape[-1]), int(sample['action'].shape[-1])
        raise RuntimeError('Could not infer cache dimensions.')

    def task_sampling_index(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for vidx, key in enumerate(self.keys):
            out.setdefault(key.task, []).append(vidx)
        return out

    def class_ids_for_vidx(self, vidx: int) -> Tuple[int, int]:
        key = self.keys[int(vidx)]
        task_id = self.task_to_id[key.task]
        task_variation_id = self.task_variation_to_id[f'{key.task}:{int(key.variation)}']
        return int(task_id), int(task_variation_id)
