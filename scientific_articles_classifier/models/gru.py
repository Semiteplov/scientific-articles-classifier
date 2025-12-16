import torch
from omegaconf import DictConfig
from torch import nn

from .base import TextClassifier


class GRUClassifier(TextClassifier):
    def __init__(self, cfg: DictConfig, num_classes: int) -> None:
        super().__init__(cfg, num_classes)

        self.embedding = nn.Embedding(
            num_embeddings=cfg.model.num_embeddings,
            embedding_dim=cfg.model.embedding_dim,
            padding_idx=0,
        )

        self.gru = nn.GRU(
            input_size=cfg.model.embedding_dim,
            hidden_size=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            batch_first=True,
            dropout=cfg.model.dropout if cfg.model.num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.classifier = nn.Linear(cfg.model.hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)
