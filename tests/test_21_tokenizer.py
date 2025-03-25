from tinynlp.utils.tokenizer import RegExTokenizer

TEST_STRING = """
Take five gallons of ale, and a large cock, the older the better. Parboil the
cock, flay him, and stamp him in a stone mortar till his bones are broken (you
must craw and gut him when you flay him), then put the cock into one quart of
sack, and put to it one and one-half pounds of raisins of the sun stoned, some
blades of mace, and a few cloves. Put all these into a canvas bag, and a little
before you find the ale has done working, put the ale and bag together into a
vessel. In a week or nine days’ time bottle it up; fill the bottle but just
above the neck, and give it the same time to ripen as other ale.
"""

REGEX_CORRECT = """Take five gallons of ale , and a large cock , the older the
better. Parboil the cock , flay him , and stamp him in a stone mortar till his
bones are broken ( you must craw and gut him when you flay him ) , then put the
cock into one quart of sack , and put to it one and one-half pounds of raisins
of the sun stoned , some blades of mace , and a few cloves. Put all these into
a canvas bag , and a little before you find the ale has done working , put the
ale and bag together into a vessel. In a week or nine days’ time bottle it up ;
fill the bottle but just above the neck , and give it the same time to ripen as
other ale .""".split()


def test_regex_tokenizer():
    assert RegExTokenizer().tokenize(TEST_STRING) == REGEX_CORRECT
