"""Stratified-by-year subsample builder for ReID datasets."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def build_subsample(
    train_csv: str | Path,
    num_ids: int,
    min_per_id: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build a stratified-by-year subsample from a training CSV.

    Logic:
      1. Read CSV (columns: image, id, pid, year, timestamp).
      2. Filter IDs that have >= min_per_id rows.
      3. Determine each ID's first-appearance year (min timestamp).
      4. Bucket IDs by first-appearance year.
      5. Distribute num_ids evenly across years (round-robin extras).
      6. Shuffle each year bucket with Random(seed), pick IDs.
      7. Return ALL rows for chosen IDs.
    """
    df = pd.read_csv(train_csv)

    # Filter IDs with >= min_per_id rows
    id_counts = df.groupby("id").size()
    valid_ids = set(id_counts[id_counts >= min_per_id].index)
    df_valid = df[df["id"].isin(valid_ids)]

    # First-appearance year per ID (by min timestamp)
    first_ts = df_valid.groupby("id")["timestamp"].min()
    first_year = df_valid.groupby("id")["year"].first()  # fallback
    # Build a proper mapping: parse timestamps to get the year of earliest record
    id_first_year: dict[str, int] = {}
    for id_ in valid_ids:
        id_rows = df_valid[df_valid["id"] == id_]
        earliest_idx = id_rows["timestamp"].min()
        year_of_earliest = id_rows.loc[
            id_rows["timestamp"] == earliest_idx, "year"
        ].iloc[0]
        id_first_year[id_] = int(year_of_earliest)

    # Bucket IDs by year
    year_buckets: dict[int, list[str]] = {}
    for id_, year in id_first_year.items():
        year_buckets.setdefault(year, []).append(id_)

    # Sort years for deterministic ordering
    sorted_years = sorted(year_buckets.keys())

    # Shuffle each bucket
    rng = random.Random(seed)
    for year in sorted_years:
        rng.shuffle(year_buckets[year])

    # Distribute num_ids evenly across years, round-robin extras
    n_years = len(sorted_years)
    base = num_ids // n_years
    extra = num_ids % n_years
    picks_per_year: dict[int, int] = {}
    for i, year in enumerate(sorted_years):
        picks_per_year[year] = base + (1 if i < extra else 0)

    # Pick IDs from each bucket
    chosen_ids: set[str] = set()
    for year in sorted_years:
        n = picks_per_year[year]
        chosen_ids.update(year_buckets[year][:n])

    # Return ALL rows for chosen IDs
    result_df = df[df["id"].isin(chosen_ids)]
    return result_df.to_dict(orient="records")


def save_subsample(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write subsample rows to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)


def load_subsample(path: str | Path) -> list[dict[str, Any]]:
    """Read subsample rows from a JSON file."""
    with open(path) as f:
        return json.load(f)
