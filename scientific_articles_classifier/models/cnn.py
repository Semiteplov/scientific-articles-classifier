import torch
from base import TextClassifier
from omegaconf import DictConfig
from torch import nn


class CNNClassifier(TextClassifier):
    def __init__(self, cfg: DictConfig, num_classes: int) -> None:
        super().__init__(cfg, num_classes)

        self.embedding = nn.Embedding(
            num_embeddings=cfg.model.num_embeddings,
            embedding_dim=cfg.model.embedding_dim,
            padding_idx=0,
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=cfg.model.embedding_dim,
                    out_channels=cfg.model.num_filters,
                    kernel_size=k,
                )
                for k in cfg.model.kernel_sizes
            ]
        )

        self.classifier = nn.Linear(
            cfg.model.num_filters * len(cfg.model.kernel_sizes),
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)

        conv_outs = [torch.max(torch.relu(conv(emb)), dim=2).values for conv in self.convs]

        features = torch.cat(conv_outs, dim=1)
        return self.classifier(features)
