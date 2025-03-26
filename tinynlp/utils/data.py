from csv import DictReader
from pathlib import Path
from typing import Iterable


class CategorizedCorpus:
    # TODO: rewrite as object composition for each type instead
    def __init__(self, data_path: str | Path):
        if isinstance(data_path, str):
            data_path = Path(data_path)

        if data_path.is_dir():
            self.data = self._process_subdir_fmt(data_path)
        elif data_path.suffix == ".tsv":
            self.data = self._process_tsv(data_path)
        else:
            self.data = self._process_plain(data_path)

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
        """returns iterator for tsv file with labels"""
        with open(data, newline="") as f:
            for line in DictReader(f, delimiter="\t", fieldnames=["text", "label"]):
                yield (str(line["text"]), int(line["label"]))

    def _process_plain(self, data: Path) -> Iterable[tuple[str, None]]:
        """returns iterator for file without labels"""
        with open(data) as f:
            for line in f:
                yield (line.strip(), None)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data)
