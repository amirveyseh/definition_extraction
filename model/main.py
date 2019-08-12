import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class Main(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim*2, opt['num_class'])
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

    def forward(self, inputs, masks, terms, defs):
        gcn_outputs, forward_outputs, backward_outputs = inputs  # unpack

        outputs = torch.cat([forward_outputs, backward_outputs], dim=1)

        logits = self.classifier(torch.cat([outputs, gcn_outputs], dim=2))

        pool_type = self.opt['pooling']
        out = pool(outputs, masks.unsqueeze(2), type=pool_type)
        out = self.out_mlp(out)
        sent_logits = self.sent_classifier(out)

        terms_out = pool(F.softmax(outputs), terms.unsqueeze(2).byte(), type=pool_type)
        defs_out = pool(F.softmax(outputs), defs.unsqueeze(2).byte(), type=pool_type)
        term_def = (terms_out * defs_out).sum(1).mean()
        not_term_def = (terms_out * defs_out[torch.randperm(terms_out.shape[0])]).sum(1).mean()

        selections = self.selector(gcn_outputs)

        return logits, sent_logits.squeeze(), selections.squeeze(), term_def, not_term_def

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