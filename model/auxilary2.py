import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class Auxilary2(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])

        self.opt = opt

    def forward(self, inputs, masks, terms, defs):
        gcn_outputs, forward_outputs, backward_outputs = inputs  # unpack

        logits = self.classifier(forward_outputs)

        return logits