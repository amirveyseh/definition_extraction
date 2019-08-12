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

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

NEGATIVE_LABEL = 'O'
SENT_NEGATIVE_LABEL = 'none'

# LABEL_TO_ID = {'O': 1, 'B-Definition': 2, 'I-Definition': 3, 'B-Ordered-Term': 4, 'B-Term': 5, 'I-Term': 6, 'I-Ordered-Term': 7, 'B-Alias-Term': 8, 'I-Alias-Term': 9, 'B-Qualifier': 10, 'I-Qualifier': 11, 'B-Ordered-Definition': 12, 'B-Secondary-Definition': 13, 'I-Secondary-Definition': 14, 'B-Referential-Definition': 15, 'I-Referential-Definition': 16, 'I-Ordered-Definition': 17, 'B-Referential-Term': 18, 'I-Referential-Term': 19}
LABEL_TO_ID = {'O': 1, 'B-Term': 2, 'B-Definition': 3, 'B-Qualifier': 4, 'I-Term': 5, 'I-Qualifier': 6, 'I-Definition': 7}
SENT_LABEL_TO_ID = {'none': 0, 'definition': 1}

INFINITY_NUMBER = 1e12
eps = 1e-10