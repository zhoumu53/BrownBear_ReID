"""YAML config loader supporting `_base_` inheritance and deep merge."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on leaves.
    Returns a new dict; does not mutate inputs.
    """
    out: dict[str, Any] = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str) -> dict:
    """Load a YAML config; if it has `_base_: <relative-path>`, recursively
    load the base and deep-merge the current file on top.
    """
    p = Path(path).resolve()
    with p.open() as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        base_path = (p.parent / base).resolve()
        base_cfg = load_config(str(base_path))
        cfg = deep_merge(base_cfg, cfg)
    return cfg
