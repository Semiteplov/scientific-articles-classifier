from collections.abc import Iterable

import torch
from mlflow.pytorch import load_model
from omegaconf import DictConfig

from scientific_articles_classifier.data import ArxivDataModule
from scientific_articles_classifier.models.cnn import CNNClassifier
from scientific_articles_classifier.models.gru import GRUClassifier
from scientific_articles_classifier.models.transformer import TransformerClassifier


def infer(cfg: DictConfig, texts: Iterable[str]) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model_uri = f"runs:/{cfg.infer.run_id}/model"
    model = load_model(model_uri)
    model.to(device)
    model.eval()

    predictions: list[str] = []

    with torch.no_grad():
        for text in texts:
            encoded = datamodule.encoder.encode(text)
            encoded = encoded.unsqueeze(0).to(device)

            logits = model(encoded)
            pred_id = logits.argmax(dim=1).item()
            predictions.append(datamodule.id_to_label[pred_id])

    return predictions
