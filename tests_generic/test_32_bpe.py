import json
import tempfile
from pathlib import Path

from tinlp.utils.tokenizer import BPETokenizer

OUT_FN = Path(tempfile.NamedTemporaryFile(delete=False).name)
TEST_FN = Path("./data/bpe_test.json")

SCD_OUT_FN = Path(tempfile.NamedTemporaryFile(delete=False).name)
SCD_TRAIN_FN = Path("./data/bpe_scd_train.txt")

TEXT = """Uh, well, Lunchables... I guess they’re, like, a thing? They have...
food in them, I think. Like, cheese, crackers, and—uh—sometimes meat? It's, uh,
kind of convenient? I guess some people really like them... but, yeah, I mean,
if you're into that kind of thing, it’s, well... probably
fine?""".replace("\n", " ")
# Assumes a vocab size of 150
TEXT_TOKENIZED = """U h ,_ we ll ,_ <unk> u n c h a b l e s .. ._ I_ gu e s s_
th e y ’ r e ,_ li k e ,_ a _ th in g ?_ <unk> h e y_ h a v e .. ._ f o o d_ in
_ th e m ,_ I_ th in k ._ <unk> i k e ,_ c h ee s e ,_ c r a c k e r s ,_ an d
<unk> u h <unk> s om e ti m e s_ m ea t ?_ I t <unk> s ,_ u h ,_ k in d_ of _ c
o n v e n i e n t ?_ I_ gu e s s_ s om e_ pe op l e_ r ea ll y_ li k e_ th e m
.. ._ bu t ,_ y ea h ,_ I_ m ea n ,_ i f_ y ou <unk> r e_ in to _ th a t_ k in
d_ of _ th in g ,_ i t’ s ,_ we ll .. ._ p r o b a b l y_ f in e ? _
""".replace("\n", " ")

UNK_TEXT = "道可道非常道"
UNK_TEXT_TOKENIZED = "<unk> <unk> <unk> <unk> <unk> <unk> _"

LRG_TRAIN_FN = Path("./data/homemade-wines.txt")
LRG_TEST_FN = Path("./data/bpe_lrg_test.json")
LRG_OUT_FN = Path(tempfile.NamedTemporaryFile(delete=False).name)


def test_BPE_tokenizer_jurafsky():
    tokenizer = BPETokenizer(data=Path("./data/bpe_train.txt"), seed=42)
    tokenizer.train(out=OUT_FN, vocab_size=19)
    with open(OUT_FN) as OUT, open(TEST_FN) as TEST:
        vocab = json.load(OUT)
        assert vocab
        correct_vocab = json.load(TEST)
    assert set(vocab) == set(correct_vocab)


def test_BPE_tokenizer_load():
    """tests training and loading"""
    BPETokenizer(data=SCD_TRAIN_FN, seed=42).train(out=SCD_OUT_FN, vocab_size=50)
    BPETokenizer(model=SCD_OUT_FN)


def test_BPE_tokenizer_tokenize_fn():
    """tests training and using the tokenizer function"""
    BPETokenizer(data=SCD_TRAIN_FN, seed=42).train(out=SCD_OUT_FN, vocab_size=150)
    tokenizer = BPETokenizer(model=SCD_OUT_FN)
    y_hat = " ".join(tokenizer.tokenize(TEXT))
    for i in range(len(y_hat)):
        assert y_hat[i] == TEXT_TOKENIZED[i]


def test_BPE_tokenizer_unk():
    """tests BPE behavior with unknown tokens"""
    # vocab size is redundant here since I want to test edge cases
    BPETokenizer(data=SCD_TRAIN_FN, seed=42).train(out=SCD_OUT_FN, vocab_size=5)
    tokenizer = BPETokenizer(model=SCD_OUT_FN)
    assert " ".join(tokenizer.tokenize(UNK_TEXT)) == UNK_TEXT_TOKENIZED


def test_BPE_tokenizer_lrg():
    """tests BPE behavior when sampling larger file"""
    BPETokenizer(data=LRG_TRAIN_FN, seed=42).train(
        out=LRG_OUT_FN,
        vocab_size=100,
        k=300,
    )
    with open(LRG_TEST_FN) as TEST_F, open(LRG_OUT_FN) as OUT_F:
        assert json.load(TEST_F) == json.load(OUT_F)


# TODO: add bpe_decode test
