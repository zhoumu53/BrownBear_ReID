"""CSV-based heads dataset for ReID training and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CsvHeadsDataset(Dataset):
    """Dataset that reads image paths and labels from a CSV file.

    The CSV must contain at least columns: image, pid, year.
    Images are loaded from ``storage_root / row["image"]``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        storage_root: str | Path,
        transform=None,
        subsample_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.transform = transform

        df = pd.read_csv(csv_path)

        # If subsample_rows provided, filter to matching images
        if subsample_rows is not None:
            keep_images = {row["image"] for row in subsample_rows}
            df = df[df["image"].isin(keep_images)].reset_index(drop=True)

        # Remap pid to contiguous range [0, num_classes)
        unique_pids = sorted(df["pid"].unique())
        self._pid_map: dict[Any, int] = {pid: idx for idx, pid in enumerate(unique_pids)}
        self.num_classes: int = len(unique_pids)

        # Store records as list of dicts for fast indexing
        self._records: list[dict[str, Any]] = df.to_dict("records")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._records[idx]
        img_path = self.storage_root / row["image"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "pid": self._pid_map[row["pid"]],
            "year": int(row["year"]),
            "image_path": str(img_path),
        }
