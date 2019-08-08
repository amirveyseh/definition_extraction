"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils


class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim*3, opt['num_class'])
        self.selector = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

        in_dim = opt['hidden_dim']
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        self.sent_classifier = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs, orig_idx2):
        _, masks, _, _, terms, defs, _, mask_definitions, _ = inputs  # unpack

        pool_type = self.opt['pooling']

        outputs, gcn_outputs, def_outputs = self.gcn_model(inputs)
        def_outputs = pool(def_outputs, mask_definitions.unsqueeze(2), type=pool_type)
        def_outputs = def_outputs[orig_idx2].repeat(1,outputs.shape[1]).view(*outputs.shape)
        logits = self.classifier(torch.cat([outputs, gcn_outputs, def_outputs], dim=2))

        out = pool(outputs, masks.unsqueeze(2), type=pool_type)
        out = self.out_mlp(out)
        sent_logits = self.sent_classifier(out)

        terms_out = pool(F.softmax(outputs), terms.unsqueeze(2).byte(), type=pool_type)
        defs_out = pool(F.softmax(outputs), defs.unsqueeze(2).byte(), type=pool_type)
        term_def = (terms_out * defs_out).sum(1).mean()
        not_term_def = (terms_out * defs_out[torch.randperm(terms_out.shape[0])]).sum(1).mean()

        wordnet_defs_out = pool(F.softmax(def_outputs), defs.unsqueeze(2).byte(), type=pool_type)
        wordnet_def = (defs_out * wordnet_defs_out).sum(1).mean()
        not_wordnet_def = (defs_out * wordnet_defs_out[torch.randperm(wordnet_defs_out.shape[0])]).sum(1).mean()

        selections = self.selector(gcn_outputs)

        return logits, sent_logits.squeeze(), selections.squeeze(), term_def, not_term_def, wordnet_def, not_wordnet_def


class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['hidden_dim'] * 2
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        # gcn output mlp layers
        in_dim = opt['hidden_dim']
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.gcn_out_mlp = nn.Sequential(*layers)

        # def output mlp layers
        in_dim = opt['hidden_dim'] * 2
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.def_out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                                              torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, head, terms, defs, _, _, adj = inputs  # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        h, pool_mask, gcn_outputs, def_outputs = self.gcn(adj, inputs)

        # pooling
        # pool_type = self.opt['pooling']
        # h_out = pool(h, pool_mask, type=pool_type)
        # outputs = torch.cat([h_out], dim=1)
        outputs = self.out_mlp(h)
        gcn_outputs = self.gcn_out_mlp(gcn_outputs)
        def_outputs = self.def_out_mlp(def_outputs)
        return outputs, gcn_outputs, def_outputs


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim']

        self.emb, self.pos_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                               dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        # def RNN
        if self.opt.get('rnn', False):
            input_size = opt['emb_dim']
            self.rnn_def = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                               dropout=opt['rnn_dropout'], bidirectional=True)
            self.rnn_drop_def = nn.Dropout(opt['rnn_dropout'])  # use on last layer output

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size, definition=False):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        if definition:
            rnn_outputs, (ht, ct) = self.rnn_def(rnn_inputs, (h0, c0))
        else:
            rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, pos, head, terms, defs, definitions, mask_definitions, adj = inputs  # unpack
        word_embs = self.emb(words)
        def_embs = self.emb(definitions)
        embs = [word_embs]
        def_embs = [def_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        embs = torch.cat(embs, dim=2)
        def_embs = torch.cat(def_embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
            def_outputs = self.rnn_drop_def(self.encode_with_rnn(def_embs, mask_definitions, definitions.size()[0], definition=True))
        else:
            gcn_inputs = embs

        lstm_outs = gcn_inputs.clone()

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW


        return lstm_outs, masks.unsqueeze(2), gcn_inputs, def_outputs


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0