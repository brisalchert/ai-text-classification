import re

def get_tokenizer():
    """
    Gets a tokenizer function for basic english sentences. Adapted from the deprecated TorchText library.
    :return: a tokenizer function for basic english splitting on whitespace.
    """

    _patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]

    _replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]

    _patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

    def _basic_english_normalize(line):
        """
        Basic normalization for a line of text in english.
        :param line: the text to normalize.
        :return: a list of tokens splitting on whitespace.
        """
        line = line.lower()
        for pattern_re, replaced_str in _patterns_dict:
            line = pattern_re.sub(replaced_str, line)
        return line.split()

    return _basic_english_normalize

class Tokenizer:
    pass