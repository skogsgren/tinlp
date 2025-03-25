from typing import Iterable
from collections import Counter
from math import log


class NGramModel:
    def __init__(self, n: int, k: float, sentences: Iterable[tuple]):
        self.n: int = n
        self.k: float = k
        self.ngrams: Counter = Counter()
        self.vocab: Counter = Counter()
        self.pad: tuple[str] = ("<s>",)
        self.end: tuple[str] = ("<e>",)

        for sentence in sentences:
            self.vocab[self.end] += 1  # we want self.end to be sentence length
            for ngram in self.get_ngrams(sentence, n=self.n):
                self.ngrams[ngram] += 1
            for word in sentence:
                self.vocab[word] += 1
            if self.n == 1:  # we don't need context for unigram
                continue
            for ngram in self.get_ngrams(sentence, n=self.n - 1):
                self.ngrams[ngram] += 1
        self.token_sum = sum(self.vocab.values())

    def get_ngrams(self, sentence: tuple, n: int) -> Iterable[tuple]:
        """given a tuple of strings returns sliding window ngram iterable"""
        s = self.pad * n + sentence + self.end
        for i in range(len(s) - n + 1):
            yield (s[i : i + n])

    def p(self, word: str, context: tuple[str]) -> float:
        """returns the probability of a word given its previous context"""
        numerator: float = self.k + self.ngrams[context + (word,)]
        # in unigram models the denominator is not the previous context but
        # rather the token size of the corpus (along with the kV smoothing)
        if self.n == 1:
            denominator: float = self.k * len(self.vocab) + self.token_sum
        else:
            denominator: float = self.k * len(self.vocab) + self.ngrams[context]
        return numerator / denominator

    def score(self, sentence: tuple) -> float:
        """given a sentence as a tuple, return the log prob of that sentence
        given model"""
        context = list(self.get_ngrams(sentence, n=self.n - 1))
        # we use log first at this step since its at the sum where the
        # floating point imprecision is starting to become a problem
        prob = [
            log(self.p(word, context[i]))
            # we have to add end here since we iterate over the sentence
            for i, word in enumerate(sentence + self.end)
        ]
        return sum(prob)
