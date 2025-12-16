from collections.abc import Mapping
from pathlib import Path

import lightning as L
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.pytorch import log_model as mlflow_log_model
from omegaconf import DictConfig, OmegaConf

from data.io import pull_data
from scientific_articles_classifier.data import ArxivDataModule

from .models import CNNClassifier, GRUClassifier, TransformerClassifier


def _cfg_to_flat_dict(cfg_section) -> dict:
    container = OmegaConf.to_container(cfg_section, resolve=True)
    if not isinstance(container, Mapping):
        raise TypeError(f"Expected dict-like config, got {type(container)}")
    return dict(container)


def train(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    pull_data([Path("data/processed/arxiv-ml.parquet")])

    datamodule = ArxivDataModule(cfg)
    datamodule.setup("fit")

    if cfg.model.name == "gru":
        model = GRUClassifier(cfg, datamodule.num_classes)
    elif cfg.model.name == "cnn":
        model = CNNClassifier(cfg, datamodule.num_classes)
    elif cfg.model.name == "transformer":
        model = TransformerClassifier(cfg, datamodule.num_classes)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    mlflow.set_tracking_uri(cfg.logging.tracking_uri)

    logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        run_name=cfg.logging.run_name,
    )

    model_params = _cfg_to_flat_dict(cfg.model)
    data_params = _cfg_to_flat_dict(cfg.data)

    logger.log_hyperparams(
        {
            "model": cfg.model.name,
            **model_params,
            **data_params,
        }
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)

    mlflow_log_model(model, artifact_path="model")
