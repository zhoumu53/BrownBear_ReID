"""CLI to build a stratified-by-year subsample JSON for fast iteration.

Usage:
    python tools/build_subsample.py \
        --train-csv /path/to/train_iid.csv \
        --num-ids 40 --min-per-id 8 \
        --output configs/data/subsample_40ids.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from project.datasets.subsample import build_subsample, save_subsample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--num-ids", type=int, default=40)
    parser.add_argument("--min-per-id", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_subsample(
        train_csv=args.train_csv,
        num_ids=args.num_ids,
        min_per_id=args.min_per_id,
        seed=args.seed,
    )
    save_subsample(rows, args.output)
    n_ids = len({r["id"] for r in rows})
    print(f"Wrote {args.output}: {len(rows)} rows, {n_ids} unique IDs")


if __name__ == "__main__":
    main()
