"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
from collections import Counter

from utils import constant, helper, vocab

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.sent_label2id = constant.SENT_LABEL_TO_ID

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.sent_id2label = dict([(v,k) for k,v in self.sent_label2id.items()])
        self.labels = [[self.id2label[l]] for d in data for l in d[-2]]
        self.sent_labels = [self.sent_id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = list(d['tokens'])
            definitions = list(d['definition'])
            if len(definitions) == 0:
                definitions = ['.']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
                definitions = [t.lower() for t in definitions]
            tokens = map_to_ids(tokens, vocab.word2id)
            definitions = map_to_ids(definitions, vocab.word2id)
            pos = map_to_ids(d['pos'], constant.POS_TO_ID)
            head = [int(x) for x in d['heads']]
            assert any([x == -1 for x in head])
            labels = [self.label2id[l] for l in d['labels']]
            dep_path = [0]*len(d['tokens'])
            for i in d['dep_path']:
                if i != -1:
                    dep_path[i] = 1
            adj = np.zeros((len(d['heads']),len(d['heads'])))
            for i, h in enumerate(d['heads']):
                    adj[i][h] = 1
                    adj[h][i] = 1
            counter = Counter(d['labels'])
            terms = [0] * len(d['labels'])
            defs = [0] * len(d['labels'])
            if counter['B-Term'] == 1 and counter['B-Definition'] == 1:
                for i, l in enumerate(d['labels']):
                    if 'Term' in l:
                        terms[i] = 1
                    if 'Definition' in l:
                        defs[i] = 1
            if self.opt['only_label'] == 1 and not self.eval:
                if d['label'] != 'none':
                    processed += [(tokens, pos, head, terms, defs, definitions, dep_path, adj, labels, self.sent_label2id[d['label']])]
            else:
                processed += [(tokens, pos, head, terms, defs, definitions, dep_path, adj, labels, self.sent_label2id[d['label']])]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def sent_gold(self):
        return self.sent_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 10

        # sort all fields by lens for easy RNN operations
        lens = [len(batch[0][i]) for i in range(len(batch[0]))]
        batch, orig_idx = sort_all(batch, lens)
        lens2 = [len(batch[5][i]) for i in range(len(batch[5]))]
        batch2, orig_idx2 = sort_all(batch, lens2)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        head = get_long_tensor(batch[2], batch_size)
        terms = get_long_tensor(batch[3], batch_size)
        defs = get_long_tensor(batch[4], batch_size)
        definitions = get_long_tensor(batch2[5], batch_size)
        mask_definitions = torch.eq(definitions, 0)
        dep_path = get_long_tensor(batch[6], batch_size).float()
        adj = get_float_tensor2D(batch[7], batch_size)

        labels = get_long_tensor(batch[8], batch_size)

        sent_labels = torch.FloatTensor(batch[9])

        return (words, masks, pos, head, terms, defs, definitions, mask_definitions, adj, labels, sent_labels, dep_path, orig_idx, orig_idx2)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor2D(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s), :len(s)] = torch.FloatTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]
