from abc import abstractmethod
from typing import Iterable

from pathlib import Path
import re


class Tokenizer:
    def __init__(self, data: None | Iterable[str] = None):
        self.data = data

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError


class RegExTokenizer(Tokenizer):
    """simple regex tokenizer which makes sure e.g. 'word.' turns into ['word',
    '.'] or 'it's' turns into ['it', ''s', '.']. I'm using a simplified version
    of the PENN treebank SED rules:
    https://web.archive.org/web/20070112180026/https://www.cis.upenn.edu/~treebank/tokenizer.sed"""

    def tokenize(self, text: str) -> list[str]:
        final_string = text
        SPEC = [
            (r'^"', r"`` "),
            (r'([ ([{<])"', r"\1 `` "),
            (r"\.\.\.", r" ... "),
            (r"[,;:@#$%&]", r" \g<0> "),
            (r'([^.])([.])([])}>"\']*)\s*$', r"\1 \2\3 "),
            (r"[?!]", r" \g<0> "),
            (r"[][(){}<>]", r" \g<0> "),
            (r"--", r" -- "),
            (r'"', r" '' "),
            (r"([^'])' ", r"\1 ' "),
            (r"  *", r" "),
        ]
        for pattern, repl in SPEC:
            final_string = re.sub(pattern, repl, final_string)
        return final_string.strip().split()


class BPETokenizer(Tokenizer):
    def __init__(self, data: None | Iterable[str], model: str | Path | None):
        super().__init__(data)
        self.model = model

    def tokenize(self, text: str) -> list[str]:
        pass

    def train(self):
        pass


def load_tokenizer(name: str | tuple) -> Tokenizer:
    if isinstance(name, tuple):
        if name[0].lower() == "bpe":
            return BPETokenizer(name[0], name[1])
    return RegExTokenizer()
