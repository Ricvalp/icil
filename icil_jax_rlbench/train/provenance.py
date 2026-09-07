from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping

import jax


def _git(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return 'not-installed'


def config_to_dict(config: Any) -> Any:
    if hasattr(config, 'to_dict'):
        return config.to_dict()
    if isinstance(config, Mapping):
        return {str(key): config_to_dict(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [config_to_dict(value) for value in config]
    return config


def collect_experiment_provenance(
    *,
    repo_root: str | Path,
    config: Any,
    experiment_id: str,
    dataset: Mapping[str, Any],
    parent_checkpoint: str = '',
    adaptation_mode: str,
    reset_policy: str,
) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    dirty_status = _git(['status', '--short'], root)
    devices = [
        {
            'id': int(device.id),
            'platform': str(device.platform),
            'device_kind': str(device.device_kind),
        }
        for device in jax.devices()
    ]
    environment_names = (
        'CUDA_VISIBLE_DEVICES',
        'JAX_PLATFORMS',
        'JAX_PLATFORM_NAME',
        'XLA_FLAGS',
        'XLA_PYTHON_CLIENT_PREALLOCATE',
    )
    return {
        'experiment_id': str(experiment_id),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git': {
            'commit': _git(['rev-parse', 'HEAD'], root),
            'branch': _git(['branch', '--show-current'], root),
            'dirty': bool(dirty_status),
            'dirty_status': dirty_status.splitlines(),
        },
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
            'packages': {
                name: _version(name)
                for name in ('jax', 'jaxlib', 'flax', 'optax', 'ml-collections', 'numpy')
            },
            'devices': devices,
            'environment': {
                name: os.environ.get(name, '') for name in environment_names
            },
        },
        'dataset': dict(dataset),
        'parent_checkpoint': str(parent_checkpoint),
        'adaptation_mode': str(adaptation_mode),
        'reset_policy': str(reset_policy),
        'resolved_config': config_to_dict(config),
    }


def write_experiment_ledger(
    run_dir: str | Path,
    *,
    config: Any,
    provenance: Mapping[str, Any],
) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / 'resolved_config.json').open('w', encoding='utf-8') as handle:
        json.dump(config_to_dict(config), handle, indent=2, sort_keys=True)
        handle.write('\n')
    with (path / 'provenance.json').open('w', encoding='utf-8') as handle:
        json.dump(dict(provenance), handle, indent=2, sort_keys=True)
        handle.write('\n')
