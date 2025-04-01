import json
import tempfile
from pathlib import Path

from tinlp.utils.tokenizer import BPETokenizer

OUT_FN = Path(tempfile.NamedTemporaryFile(delete=False).name)
TEST_FN = Path("./data/bpe_test.json")


def test_BPE_tokenizer_jurafsky():
    tokenizer = BPETokenizer(Path("./data/bpe_train.txt"), seed=42)
    tokenizer.train(out=OUT_FN, vocab_size=19)
    with open(OUT_FN) as OUT, open(TEST_FN) as TEST:
        vocab = json.load(OUT)
        correct_vocab = json.load(TEST)
    assert set(vocab) == set(correct_vocab)
