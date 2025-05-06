import tempfile
from pathlib import Path

import pytest
import tinlp
from tinlp.nn import NGram, get_ngram_tensors
from tinlp.utils.data import CorpusPlain
from tinlp.utils.text import search_vectors
from tinlp.utils.tokenizer import load_tokenizer

VECTOR_OUT_TMP = Path(
    tempfile.NamedTemporaryFile(delete=False, suffix=".vector.gz").name
)
TRAIN_DATA = Path("./data/bengio/english-train.txt.gz")
TEST_DATA = Path("./data/bengio/english-test-1k.txt.gz")
NGRAM_SIZE = 4
SEED = 42
N_EPOCHS = 2

DEVICE = "cuda"

TOKENIZER = "regex"
PARAMS = {
    "batch_size": 1024,
    "hid_size": 80,
    "ngram_size": NGRAM_SIZE,
    "emb_size": 100,
    "grad_clip": 1.0,
    "lr": 0.001,
    "seed": SEED,
    "pbar": True,
    "device": DEVICE,
}

print(f"{VECTOR_OUT_TMP=}")


@pytest.mark.skipif(not (hasattr(tinlp, "nn")), reason="requires nn plugins.")
def test_nn_ngram():
    import torch

    X, y, vocab = get_ngram_tensors(
        corpus=CorpusPlain(TRAIN_DATA),
        tokenizer=load_tokenizer(TOKENIZER),
        n=NGRAM_SIZE,
        min_occ=5,
        device=DEVICE,
    )

    # inv_vocab = {v: k for k, v in vocab.items()}
    # def tensor_to_tokens(x):
    #     return [inv_vocab[n] for n in x.cpu().numpy()]

    # print("X[2]", X[2], tensor_to_tokens(X[2]))
    # print("y[2]", y[2], inv_vocab[int(y[2].cpu())])
    # exit(1)

    # i.e. making sure unk token id is not in target arr
    y_arr = y.cpu().numpy()
    assert 1 not in y_arr

    X_test, y_test, _ = get_ngram_tensors(
        CorpusPlain(TEST_DATA),
        tokenizer=load_tokenizer(TOKENIZER),
        n=NGRAM_SIZE,
        vocab=vocab,
        device=DEVICE,
    )

    params = PARAMS
    params["vocab_size"] = len(vocab)
    params["steps"] = (X.shape[0] // params["batch_size"]) * N_EPOCHS
    model = NGram(**params)
    print(model)

    loss_before = torch.nn.functional.cross_entropy(model(X_test), y_test).item()
    print(f"{loss_before=}")
    # assert loss_before > 10.0

    model.fit(X, y)
    # to avoid CUDA overflow
    X.detach()
    y.detach()
    del X
    del y

    loss_after = torch.nn.functional.cross_entropy(model(X_test), y_test).item()
    # assert loss_after < 5.5
    print(loss_before, loss_after)

    print("exporting vectors")
    model.export_vectors(vocab, VECTOR_OUT_TMP)
    top_k = search_vectors(VECTOR_OUT_TMP, "friday", k=20)
    print(top_k)


test_nn_ngram()
