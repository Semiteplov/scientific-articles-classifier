import torch
from omegaconf import DictConfig
from torch import nn

from .base import TextClassifier


class TransformerClassifier(TextClassifier):
    def __init__(self, cfg: DictConfig, num_classes: int) -> None:
        super().__init__(cfg, num_classes)

        self.embedding = nn.Embedding(
            num_embeddings=cfg.model.num_embeddings,
            embedding_dim=cfg.model.embedding_dim,
            padding_idx=0,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.embedding_dim,
            nhead=cfg.model.num_heads,
            dim_feedforward=cfg.model.ff_dim,
            dropout=cfg.model.dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.model.num_layers,
        )

        self.classifier = nn.Linear(cfg.model.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        enc = self.encoder(emb)
        pooled = enc.mean(dim=1)
        return self.classifier(pooled)
