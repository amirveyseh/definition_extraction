import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class Discriminator(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.embed = Variable(torch.from_numpy(np.random.rand(opt['num_class'], opt['label_emb']))).float().cuda()

        input_size = opt['label_emb']
        self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                dropout=opt['rnn_dropout'], bidirectional=True)
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])

        in_dim = opt['hidden_dim']*2
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        in_dim = opt['hidden_dim']
        self.disc = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, labels, masks):
        label_emb = labels.matmul(self.embed)

        label_outputs = self.rnn_drop(self.encode_with_rnn(label_emb, masks, labels.size()[0]))

        pool_type = self.opt['pooling']
        label_outputs = pool(label_outputs, masks.unsqueeze(2), type=pool_type)

        outputs = self.out_mlp(label_outputs)

        return self.disc(outputs)



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