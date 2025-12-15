import subprocess
from collections.abc import Iterable
from pathlib import Path


def pull_data(targets: Iterable[Path]) -> None:
    args = ["dvc", "pull", *[str(t) for t in targets]]
    subprocess.run(args, check=True)
