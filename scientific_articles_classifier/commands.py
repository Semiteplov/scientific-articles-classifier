from collections.abc import Sequence

from hydra import compose, initialize
from omegaconf import DictConfig

from scientific_articles_classifier.infer import infer
from scientific_articles_classifier.train import train


def _parse_models(models: str | Sequence[str]) -> list[str]:
    if isinstance(models, str):
        return [m.strip() for m in models.split(",") if m.strip()]
    return [str(m).strip() for m in models if str(m).strip()]


def train_cmd(models: str | Sequence[str] = "gru", overrides: Sequence[str] | None = None) -> None:
    overrides_list = list(overrides) if overrides is not None else []
    model_names = _parse_models(models)

    for model_name in model_names:
        with initialize(config_path="../configs", version_base=None):
            cfg: DictConfig = compose(
                config_name="config",
                overrides=[f"model={model_name}", *overrides_list],
            )
        train(cfg)


def infer_cmd(
    texts: list[str],
    run_id: str,
    overrides: Sequence[str] | None = None,
) -> None:
    overrides_list = list(overrides) if overrides is not None else []

    with initialize(config_path="../configs", version_base=None):
        cfg: DictConfig = compose(config_name="config", overrides=overrides_list)

    cfg.infer.run_id = run_id
    preds = infer(cfg, texts)

    for text, pred in zip(texts, preds, strict=False):
        print(f"[{pred}] {text[:80]}")
