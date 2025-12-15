from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ArxivDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels.astype(np.int64)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def _encode(self, text: str) -> torch.Tensor:
        encoded = text.encode("utf-8", errors="ignore")[: self.max_length]
        padded = encoded + b"\x00" * max(0, self.max_length - len(encoded))
        return torch.tensor(list(padded), dtype=torch.long)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(self.texts[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class ArxivDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

    @property
    def num_classes(self) -> int:
        return len(self.label_to_id)

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_parquet(Path(self.cfg.data.processed_path))

        required = {"abstract", "label"}
        if self.cfg.data.text.use_title:
            required.add("title")

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        if self.cfg.data.text.use_title:
            texts = (df["title"].fillna("") + "\n\n" + df["abstract"].fillna("")).tolist()
        else:
            texts = df["abstract"].fillna("").tolist()

        labels_str = df["label"].astype(str).tolist()
        unique_labels = sorted(set(labels_str))

        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        labels = np.array([self.label_to_id[label] for label in labels_str], dtype=np.int64)

        train_idx, val_idx = train_test_split(
            np.arange(len(texts)),
            test_size=self.cfg.data.split.val_size,
            random_state=self.cfg.data.split.seed,
            stratify=labels,
        )

        self.train_dataset = ArxivDataset(
            [texts[i] for i in train_idx],
            labels[train_idx],
            self.cfg.data.text.max_length,
        )

        self.val_dataset = ArxivDataset(
            [texts[i] for i in val_idx],
            labels[val_idx],
            self.cfg.data.text.max_length,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before dataloaders")

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.loader.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.loader.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before dataloaders")

        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.loader.num_workers,
            pin_memory=True,
        )
