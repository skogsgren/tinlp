import tempfile
from pathlib import Path

from tinlp.lm import NGramNN
from tinlp.utils.data import CorpusPlain
from tinlp.utils.tokenizer import load_tokenizer
from tinlp.utils.text import search_vectors

OUT = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".vector.gz").name)
TEST_DATA = Path("./data/bengio/english-train.txt.gz")
NGRAM_SIZE = 4
SEED = 42

TOKENIZER = "regex"
PARAMS = {
    "batch_size": 1024,
    "hidden_size": 80,
    "ngram_size": NGRAM_SIZE,
    "embedding_size": 100,
    "gradient_clipping_value": 1.0,
    "learning_rate": 0.001,
    "seed": SEED,
    "show_progress": True,
}


def test_nn_ngram_loss():
    X, y, vocab = CorpusPlain(TEST_DATA).get_ngram_tensors(
        tokenizer=load_tokenizer(TOKENIZER),
        n=NGRAM_SIZE,
        min_occ=5,
    )
    print(f"{len(vocab)=}")
    print(f"{X.shape=}")
    print(f"{y.shape=}")
    y_arr = y.numpy()
    assert 1 not in y_arr  # i.e. unk token id
    params = PARAMS
    params["vocab_size"] = len(vocab)
    batch_size = params["batch_size"]
    params["steps"] = (X.shape[0] // params["batch_size"]) * 2  # two epochs
    model = NGramNN(params)
    score_before_training = model.eval(X, y)
    loss_before_training = model(X[:batch_size]).cross_entropy(y[:batch_size]).item()
    print(f"{loss_before_training=}")
    model.fit(X, y)
    loss_after_training = model(X[:batch_size]).cross_entropy(y[:batch_size]).item()
    score_after_training = model.eval(X, y)
    print(loss_before_training, loss_after_training)
    print(score_before_training, score_after_training)


def test_nn_ngram_vectors():
    X, y, vocab = CorpusPlain(TEST_DATA).get_ngram_tensors(
        tokenizer=load_tokenizer(TOKENIZER),
        n=NGRAM_SIZE,
        min_occ=5,
    )
    params = PARAMS
    params["vocab_size"] = len(vocab)
    params["steps"] = (X.shape[0] // params["batch_size"]) * 2  # two epochs
    model = NGramNN(params)
    model.fit(X, y)
    model.export_vectors(vocab, OUT)

    top_k = search_vectors(OUT, "friday", k=15)
    print(top_k)


test_nn_ngram_loss()
test_nn_ngram_vectors()

""" QUESTIONS
- how can I possible load in the entire thing into memory when it just crashes if I attempt to do so?
    * even a small subset takes up 3.2GB
    * since I'm using random sampling, what I could do is to train it "in
      folds", meaning I split it in say ten folds, and then train each fold for 2 epochs.
    * this is going to be very slow though, since I have to recreate the Tensor
    for each fold, reading the file again, iterating over each line until I get
    to the folds
- why is the loss so different?
- are my topk vectors supposed to look like that?
"""
