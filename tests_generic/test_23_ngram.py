from tinlp.utils.tokenizer import RegExTokenizer
from tinlp.lm import NGramModel

FN = "data/homemade-wines.txt"
TEST_SENTENCES = [
    "Take white currants when quite ripe",
    "Take white matchsticks when quite ripe",
    "Take white currants when quite dead",
]
TEST_SENTENCES_SCORES = [-44.08373, -57.84285, -50.13748]


def sent_reader(fn):
    tokenizer = RegExTokenizer()
    with open(fn) as f:
        for line in f:
            yield tuple(tokenizer.tokenize(line))


def test_ngram_model():
    model = NGramModel(2, 0.001, sent_reader(FN))
    for sentence, score in zip(TEST_SENTENCES, TEST_SENTENCES_SCORES):
        assert round(model.score(tuple(sentence.split())), 5) == score
