from tinynlp.utils.tokenizer import RegExTokenizer
from tinynlp.lm import NGramModel

FN = "data/homemade-wines.txt"
TEST_SENTENCES = [
    "Take white currants when quite ripe",
    "Take white matchsticks when quite ripe",
    "Take white currants when quite dead",
]
TEST_SENTENCES_SCORES = [-29.84274, -43.56262, -35.76526]


def sent_reader(fn):
    tokenizer = RegExTokenizer()
    with open(fn) as f:
        for line in f:
            yield tuple(tokenizer.tokenize(line))


def test_ngram_model():
    model = NGramModel(2, 0.001, sent_reader(FN))
    for sentence, score in zip(TEST_SENTENCES, TEST_SENTENCES_SCORES):
        assert round(model.score(tuple(sentence.split())), 5) == score
