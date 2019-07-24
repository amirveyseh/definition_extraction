"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ADJ': 2, 'PRON': 3, 'AUX': 4, 'ADP': 5, 'PROPN': 6, 'VERB': 7, 'DET': 8,
             'X': 9, 'INTJ': 10, 'CCONJ': 11, 'NUM': 12, 'PUNCT': 13, 'PART': 14, 'SYM': 15, 'ADV': 16, 'NOUN': 17}

NEGATIVE_LABEL = 'O'

LABEL_TO_ID = {'B-attribute': 1, 'B-value': 2, 'O': 3, 'I-object': 4, 'I-action': 5, 'B-action': 6, 'B-object': 7,
               'I-attribute': 8, 'I-value': 9}

INFINITY_NUMBER = 1e12
