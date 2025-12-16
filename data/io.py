import subprocess
from collections.abc import Iterable
from pathlib import Path


def pull_data(targets: Iterable[Path]) -> None:
    for target in targets:
        subprocess.run(
            ["dvc", "pull", target.as_posix()],
            check=True,
        )
