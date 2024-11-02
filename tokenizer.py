import re

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def get_tokenizer():
    """
    Gets a tokenizer function for basic english sentences. Adapted from the deprecated TorchText library.
    :return: a tokenizer function for basic english splitting on whitespace.
    """

    _patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]

    _replacements = ["", "", "", " ", "", "", "", "", "", " ", " ", " "]

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
        line = remove_emoji(line)
        return line.split()

    return _basic_english_normalize

class Tokenizer:
    pass