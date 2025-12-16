from collections.abc import Sequence

from hydra import compose, initialize
from omegaconf import DictConfig

from scientific_articles_classifier.infer import infer
from scientific_articles_classifier.train import train


def train_cmd(models: Sequence[str] = ("gru",), overrides: Sequence[str] | None = None) -> None:
    overrides = list(overrides) if overrides is not None else []

    for model_name in models:
        with initialize(config_path="../configs", version_base=None):
            cfg: DictConfig = compose(
                config_name="config",
                overrides=[f"model={model_name}", *overrides],
            )

        train(cfg)


def infer_cmd(
    texts: list[str],
    run_id: str,
    overrides: Sequence[str] | None = None,
) -> None:
    overrides = list(overrides) if overrides is not None else []

    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name="config", overrides=overrides)

    cfg.infer.run_id = run_id
    preds = infer(cfg, texts)

    for text, pred in zip(texts, preds, strict=False):
        print(f"[{pred}] {text[:80]}")
