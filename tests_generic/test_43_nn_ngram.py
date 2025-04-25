import tempfile
from pathlib import Path

from tinlp.lm import NGramNN
from tinlp.utils.data import CorpusPlain
from tinlp.utils.tokenizer import BPETokenizer, load_tokenizer

OUT = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".vector.gz").name)
TEST_DATA = Path("./data/homemade-wines.txt")
NGRAM_SIZE = 4
BPE_VOCAB_SIZE = 1500
SEED = 42
PARAMS = {
    "steps": 500,
    "hidden_size": 80,
    "ngram_size": NGRAM_SIZE,
    "embedding_size": 100,
    "gradient_clipping_value": 1.0,
    "batch_size": 16,
    "learning_rate": 0.001,
    "seed": SEED,
    "show_progress": True,
}

print("TMP VECTOR BUFFER=", OUT)


def test_nn_ngram():
    X, y, vocab = CorpusPlain(TEST_DATA).get_ngram_tensors(
        tokenizer=load_tokenizer("regex"),
        n=NGRAM_SIZE,
        min_occ=5,
    )
    y_arr = y.numpy()
    assert 1 not in y_arr  # i.e. unk token id
    params = PARAMS
    params["vocab_size"] = len(vocab)
    model = NGramNN(params)
    score_before_training = model.eval(X, y)
    model.fit(X, y)
    assert model.eval(X, y) > score_before_training


def test_nn_ngram_bpe():
    bpe_tokenizer = BPETokenizer(model=Path("./data/nn_ngram_bpe.json"))
    X, y, vocab = CorpusPlain(TEST_DATA).get_ngram_tensors(
        tokenizer=bpe_tokenizer,
        n=NGRAM_SIZE,
        min_occ=5,
    )
    print(f"{len(vocab)=}")

    params = PARAMS
    params["vocab_size"] = len(vocab)
    model = NGramNN(params)
    score_before_training = model.eval(X, y)
    loss_before_training = model(X).cross_entropy(y).item()
    model.fit(X, y)
    score_after_training = model.eval(X, y)
    loss_after_training = model(X).cross_entropy(y).item()
    assert score_after_training > score_before_training

    print(f"{loss_before_training=}; {loss_after_training=}")
    print(f"{score_before_training=}; {score_after_training=}")


def test_vector_export():
    X, y, vocab = CorpusPlain(TEST_DATA).get_ngram_tensors(
        tokenizer=load_tokenizer("regex"),
        n=NGRAM_SIZE,
        min_occ=5,
    )
    params = PARAMS
    params["vocab_size"] = len(vocab)
    m = NGramNN(params)
    m.fit(X, y)
    m.export_vectors(vocab, OUT)
    assert OUT.exists()
    assert OUT.stat().st_size > 0
