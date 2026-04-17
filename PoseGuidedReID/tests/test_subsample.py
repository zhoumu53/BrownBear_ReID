"""Tests for project.datasets.subsample — stratified-by-year subsample builder."""
import json

import pandas as pd
import pytest

from project.datasets.subsample import build_subsample, load_subsample, save_subsample


@pytest.fixture
def sample_csv(tmp_path):
    rows = []
    pid = 0
    for year in [2018, 2019, 2020]:
        for n in range(4):
            id_ = f"bear_{year}_{n}"
            pid += 1
            for i in range(10):
                rows.append({
                    "image": f"{year}_heads/images/img_{id_}_{i}.JPG",
                    "id": id_, "pid": pid, "year": year,
                    "timestamp": f"{year}-06-{(i % 28) + 1:02d} 10:00:00",
                })
    df = pd.DataFrame(rows)
    p = tmp_path / "train.csv"
    df.to_csv(p, index=False)
    return p


def test_build_subsample_respects_min_per_id(sample_csv):
    """Every ID in output has >= min_per_id rows."""
    rows = build_subsample(sample_csv, num_ids=6, min_per_id=5, seed=42)
    counts = {}
    for r in rows:
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    for id_, count in counts.items():
        assert count >= 5, f"{id_} has only {count} rows, expected >= 5"


def test_build_subsample_picks_requested_count(sample_csv):
    """Output has exactly num_ids distinct IDs."""
    rows = build_subsample(sample_csv, num_ids=6, min_per_id=5, seed=42)
    ids = {r["id"] for r in rows}
    assert len(ids) == 6


def test_build_subsample_stratified_across_years(sample_csv):
    """6 IDs from 3 years = 2 per year."""
    rows = build_subsample(sample_csv, num_ids=6, min_per_id=5, seed=42)
    # Determine first-appearance year for each chosen ID
    year_counts = {}
    id_year = {}
    for r in rows:
        if r["id"] not in id_year:
            id_year[r["id"]] = r["year"]
    for year in id_year.values():
        year_counts[year] = year_counts.get(year, 0) + 1
    assert year_counts == {2018: 2, 2019: 2, 2020: 2}


def test_build_subsample_deterministic(sample_csv):
    """Same seed -> same output."""
    rows1 = build_subsample(sample_csv, num_ids=6, min_per_id=5, seed=42)
    rows2 = build_subsample(sample_csv, num_ids=6, min_per_id=5, seed=42)
    assert rows1 == rows2


def test_build_subsample_different_seed_different_pick(sample_csv):
    """Different seed -> different IDs."""
    rows1 = build_subsample(sample_csv, num_ids=6, min_per_id=5, seed=42)
    rows2 = build_subsample(sample_csv, num_ids=6, min_per_id=5, seed=99)
    ids1 = {r["id"] for r in rows1}
    ids2 = {r["id"] for r in rows2}
    assert ids1 != ids2


def test_save_and_load_roundtrip(sample_csv, tmp_path):
    """save then load == original."""
    rows = build_subsample(sample_csv, num_ids=6, min_per_id=5, seed=42)
    out = tmp_path / "sub.json"
    save_subsample(rows, out)
    loaded = load_subsample(out)
    assert loaded == rows
