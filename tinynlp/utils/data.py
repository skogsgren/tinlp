from pathlib import Path
from typing import Iterable
from csv import DictReader


class CategorizedCorpus:
    def __init__(self, data_path: str | Path):
        if isinstance(data_path, str):
            data_path = Path(data_path)

        if data_path.is_dir():
            self.data = self._process_subdir_fmt(data_path)
        elif data_path.suffix == ".tsv":
            self.data = self._process_tsv(data_path)
        else:
            raise ValueError("Incorrect format provided to CategorizedCorpus")

    def _process_subdir_fmt(self, data: Path) -> Iterable[tuple[str, int]]:
        SUBDIR_LABELS = ["neg", "pos"]
        folders = sorted(
            [x for x in data.iterdir() if x.is_dir() and x.name in SUBDIR_LABELS],
            key=lambda x: x.name,  # to ensure the order is neg, pos
        )
        for label in SUBDIR_LABELS:
            if label in [x.name for x in folders]:
                continue
            raise ValueError(f"{label} not in subdirectories for {data}")
        for i, folder in enumerate(folders):
            for file in folder.iterdir():
                with open(file) as f:
                    yield (f.read().strip(), i)

    def _process_tsv(self, data: Path) -> Iterable[tuple[str, int]]:
        with open(data, newline="") as f:
            for line in DictReader(f, delimiter="\t", fieldnames=["text", "label"]):
                yield (str(line["text"]), int(line["label"]))

    def __iter__(self):
        return self.data
