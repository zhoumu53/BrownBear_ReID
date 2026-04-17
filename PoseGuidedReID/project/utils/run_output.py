"""Per-run output directory utilities.

Creates structured output directories for training runs, captures git info,
dumps frozen config, and writes metrics.
"""

import json
import subprocess
from pathlib import Path
from typing import Union


def _capture_git_info() -> dict[str, str]:
    """Capture current repo SHA, dirty status, and parent superproject SHA."""
    info: dict[str, str] = {}

    # Current repo SHA
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["sha"] = sha
    except Exception:
        info["sha"] = "unknown"

    # Dirty status
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = str(bool(status))
    except Exception:
        info["dirty"] = "unknown"

    # Current branch
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = branch
    except Exception:
        info["branch"] = "unknown"

    # Parent superproject SHA
    try:
        superproject_dir = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if superproject_dir:
            parent_sha = subprocess.check_output(
                ["git", "-C", superproject_dir, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            info["parent_sha"] = parent_sha
        else:
            info["parent_sha"] = "none (not a submodule)"
    except Exception:
        info["parent_sha"] = "unknown"

    return info


def init_run_output(output_dir: str, frozen_config: dict) -> dict[str, Path]:
    """Initialise a per-run output directory with subdirs, config dump, and git info.

    Parameters
    ----------
    output_dir : str
        Root path for the run output.
    frozen_config : dict
        Configuration dict to dump as YAML.

    Returns
    -------
    dict[str, Path]
        Keys: root, checkpoints, tensorboard, log, metrics.
    """
    root = Path(output_dir)
    checkpoints = root / "checkpoints"
    tensorboard = root / "tensorboard"

    root.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    tensorboard.mkdir(exist_ok=True)

    # Dump frozen config as YAML
    try:
        import yaml
        config_text = yaml.dump(frozen_config, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback: write as JSON if PyYAML is not available
        config_text = json.dumps(frozen_config, indent=2)
    (root / "config.yaml").write_text(config_text)

    # Capture git info
    git_info = _capture_git_info()
    lines = [f"{k}: {v}" for k, v in git_info.items()]
    (root / "git.txt").write_text("\n".join(lines) + "\n")

    return {
        "root": root,
        "checkpoints": checkpoints,
        "tensorboard": tensorboard,
        "log": root / "log.txt",
        "metrics": root / "metrics.json",
    }


def write_metrics(output_dir: Union[str, Path], metrics: dict) -> None:
    """Write metrics dict as pretty-printed JSON to output_dir/metrics.json.

    Parameters
    ----------
    output_dir : str or Path
        Run output directory (must already exist).
    metrics : dict
        Metrics to serialise.
    """
    path = Path(output_dir) / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
