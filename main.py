import fire

from scientific_articles_classifier.commands import infer_cmd, train_cmd


def main() -> None:
    fire.Fire(
        {
            "train": train_cmd,
            "infer": infer_cmd,
        }
    )


if __name__ == "__main__":
    main()
