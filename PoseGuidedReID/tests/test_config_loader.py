"""Unit tests for the YAML config loader."""
import pytest
import yaml
from pathlib import Path
from project.config.loader import load_config, deep_merge


def test_deep_merge_overrides_leaf():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}}
    assert deep_merge(base, override) == {"a": 1, "b": {"c": 99, "d": 3}}


def test_deep_merge_does_not_mutate_inputs():
    base = {"x": {"y": 1}}
    override = {"x": {"y": 2}}
    out = deep_merge(base, override)
    assert base == {"x": {"y": 1}}
    assert override == {"x": {"y": 2}}
    assert out == {"x": {"y": 2}}


def test_load_config_no_base(tmp_path):
    cfg_path = tmp_path / "leaf.yaml"
    cfg_path.write_text("a: 1\nb:\n  c: 2\n")
    out = load_config(str(cfg_path))
    assert out == {"a": 1, "b": {"c": 2}}


def test_load_config_with_base(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("a: 1\nb:\n  c: 2\n  d: 3\n")
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("_base_: base.yaml\nb:\n  c: 99\ne: hello\n")
    out = load_config(str(leaf))
    assert out == {"a": 1, "b": {"c": 99, "d": 3}, "e": "hello"}


def test_load_config_chained_base(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text("level: a\nshared: 1\n")
    b = tmp_path / "b.yaml"
    b.write_text("_base_: a.yaml\nlevel: b\nshared: 2\n")
    c = tmp_path / "c.yaml"
    c.write_text("_base_: b.yaml\nlevel: c\n")
    out = load_config(str(c))
    assert out == {"level": "c", "shared": 2}
