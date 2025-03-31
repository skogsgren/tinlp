from tinlp.utils.tokenizer import RegExTokenizer

TEST_STRING = """
Hello... wait—what?! $100.50; isn't that crazy? (Yes!) "Wow," she said: "Unbelievable."
""".replace("\n", "")

REGEX_CORRECT = [
    "hello",
    "...",
    "wait",
    "—",
    "what",
    "?!",
    "$",
    "100",
    ".",
    "50",
    ";",
    "isn",
    "'",
    "t",
    "that",
    "crazy",
    "?",
    "(",
    "yes",
    "!)",
    '"',
    "wow",
    ',"',
    "she",
    "said",
    ":",
    '"',
    "unbelievable",
    '."',
]


def test_regex_tokenizer():
    for i, token in enumerate(RegExTokenizer().tokenize(TEST_STRING)):
        assert token == REGEX_CORRECT[i]
