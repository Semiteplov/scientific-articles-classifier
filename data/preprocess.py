import gzip
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

ML_CATEGORIES = {
    "cs.LG",
    "stat.ML",
    "cs.AI",
    "cs.CL",
    "cs.CV",
}


def iter_arxiv_records(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def extract_label(categories: str) -> str | None:
    for category in categories.split():
        if category in ML_CATEGORIES:
            return category
    return None


def preprocess_arxiv(
    raw_path: Path,
    output_path: Path,
    min_abstract_length: int = 100,
) -> None:
    records: list[dict] = []

    for paper in iter_arxiv_records(raw_path):
        label = extract_label(paper.get("categories", ""))
        if label is None:
            continue

        abstract = paper.get("abstract", "").strip()
        if len(abstract) < min_abstract_length:
            continue

        records.append(
            {
                "id": paper["id"],
                "title": paper.get("title", "").strip(),
                "abstract": abstract,
                "label": label,
            }
        )

    df = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
