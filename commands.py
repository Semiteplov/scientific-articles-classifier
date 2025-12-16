import hydra
from omegaconf import DictConfig

from scientific_articles_classifier import infer, train


def train_cmd() -> None:
    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def _train(cfg: DictConfig) -> None:
        train(cfg)

    _train()


def infer_cmd(
    texts: list[str],
    run_id: str,
) -> None:
    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def _infer(cfg: DictConfig) -> None:
        cfg.infer.run_id = run_id
        preds = infer(cfg, texts)
        for text, pred in zip(texts, preds, strict=False):
            print(f"[{pred}] {text[:80]}")

    _infer()
