import fire

from commands import infer_cmd, train_cmd


def main() -> None:
    fire.Fire(
        {
            "train": train_cmd,
            "infer": infer_cmd,
        }
    )
