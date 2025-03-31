import random
from csv import DictReader
from pathlib import Path
from typing import Any, Iterable


class CategorizedCorpus:
    # TODO: rewrite as object composition for each type instead
    # TODO: check if first row contains labels first
    # TODO: URGENT: This class is turning into a mess. do something soon.
    def __init__(self, data_path: str | Path, data_type: None | str = None):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if data_type == "unimorph":
            self.data = self._process_unimorph(data_path)
        elif data_type == "conll2003":
            self.data = self._process_conll_2003(data_path)
        elif data_path.is_dir():
            self.data = self._process_subdir_fmt(data_path)
        elif data_path.suffix == ".tsv":
            self.data = self._process_tsv(data_path)
        elif data_path.suffix == ".csv":
            self.data = self._process_csv(data_path)
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

    def _process_csv(self, data: Path) -> Iterable[tuple[str, Any]]:
        """returns iterator for tsv file with labels"""
        with open(data, newline="", encoding="utf-8") as f:
            for line in DictReader(f, fieldnames=["text", "label"]):
                yield (str(line["text"]), line["label"])

    def _process_tsv(self, data: Path) -> Iterable[tuple[str, Any]]:
        """returns iterator for tsv file with labels"""
        with open(data, newline="", encoding="utf-8") as f:
            for line in DictReader(f, delimiter="\t", fieldnames=["text", "label"]):
                yield (str(line["text"]), line["label"])

    def _process_plain(self, data: Path) -> Iterable[tuple[str, None]]:
        """returns iterator for file without labels"""
        with open(data) as f:
            for line in f:
                yield (line.strip(), None)

    def _process_conll_2003(
        self, data: Path
    ) -> Iterable[tuple[tuple[tuple[str]], tuple[str]]]:
        with open(data) as f:
            seq = ([], [])
            for i, line in enumerate(x.strip().split() for x in f):
                if i == 0:
                    continue
                if not line:
                    if i == 1:
                        continue
                    yield tuple(seq[0]), tuple(seq[1])
                    seq = ([], [])
                    continue
                seq[0].append((line[0], line[1], line[2]))
                seq[1].append(line[3])

    # NOTE: this isn't needed right? Just use the TSV instead and handle labels better
    def _process_unimorph(self, data: Path) -> Iterable[tuple[str, str]]:
        """returns iterator for unimorph file"""
        with open(data) as f:
            for line in f:
                line = line.strip().split("\t")
                yield line[1], line[2]

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data)


def get_data_split(
    data_fn: Path,
    data_type: None | str = None,
    shuffle: bool = False,
    seed: int = 42,
):
    data = [x for x in CategorizedCorpus(data_fn, data_type=data_type)]
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    try:  # most cases we should have ints as labels
        return [x[0] for x in data], [int(x[1]) for x in data]
    except ValueError:  # but we can have strings, so just return it as is
        return [x[0] for x in data], [x[1] for x in data]
