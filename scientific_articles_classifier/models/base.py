from abc import abstractmethod

import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn


class TextClassifier(L.LightningModule):
    def __init__(self, cfg: DictConfig, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.model.lr,
            weight_decay=self.cfg.model.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), on_epoch=True)
        self.log("train_f1", self.train_f1(logits, y), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc(logits, y), on_epoch=True)
        self.log("val_f1", self.val_f1(logits, y), on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc(logits, y), on_epoch=True)
        self.log("test_f1", self.test_f1(logits, y), on_epoch=True)
