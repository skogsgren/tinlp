from csv import DictReader
from pathlib import Path
from typing import Iterable
from abc import abstractmethod


class Corpus:
    """abstract class for Corpus subclasses"""

    def __init__(self, data_path: str | Path, **params):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data = self._process(data_path, **params)

    @abstractmethod
    def _process(self, data: Path, **params):
        raise NotImplementedError

    def get_arrays(self, unsqueeze: bool = False) -> tuple[list, list]:
        """returns two lists, X with independent, and y with dependent variables"""
        data = [x for x in self.data]
        if unsqueeze:
            return [(x[0],) for x in data], [(x[1],) for x in data]
        return [x[0] for x in data], [x[1] for x in data]

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data)


class CorpusCSV(Corpus):
    def _process(self, data: Path, **params):
        """returns iterator for tsv file with labels"""
        delim = params.get("delimiter", ",")
        header = params.get("header", False)
        with open(data, newline="", encoding="utf-8") as f:
            if header:
                raise NotImplementedError

            fieldnames = params.get("fieldnames", ["text", "label"])
            for line in DictReader(f, delimiter=delim, fieldnames=fieldnames):
                if params.get("label_to_int", False):
                    yield (str(line["text"]), int(line["label"]))
                else:
                    yield (str(line["text"]), line["label"])


class CorpusSubDir(Corpus):
    def _process(self, data: Path, **params):
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


class CorpusCONLL2003(Corpus):
    def _process(self, data: Path, **params):
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


class CorpusPlain(Corpus):
    def _process(self, data: Path, **params) -> Iterable[str]:
        """returns iterator for file without labels"""
        with open(data) as f:
            for line in f:
                yield line.strip()
