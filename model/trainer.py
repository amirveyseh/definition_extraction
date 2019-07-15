"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchcrf import CRF

from model.gcn import GCNClassifier
from utils import constant, torch_utils

import random

random.seed(1234)


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:5]]
        labels = Variable(batch[5].cuda())
        sent_labels = Variable(batch[6].cuda())
        dep_path = Variable(batch[7].cuda())
    else:
        print("Error")
        exit(1)
    tokens = batch[0]
    head = batch[3]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, sent_labels, dep_path, tokens, head, lens


class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.crf = CRF(self.opt['num_class'], batch_first=True)
        self.bc = nn.BCELoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
            self.crf.cuda()
            self.bc.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels, sent_labels, dep_path, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, class_logits, selections, positive, negative = self.model(inputs)

        labels = labels - 1
        labels[labels < 0] = 0
        mask = inputs[1].float()
        mask[mask == 0.] = -1.
        mask[mask == 1.] = 0.
        mask[mask == -1.] = 1.
        mask = mask.byte()
        loss = -self.crf(logits, labels, mask=mask)

        sent_loss = self.bc(class_logits, sent_labels)
        loss += self.opt['sent_loss'] * sent_loss

        selection_loss = self.bc(selections.view(-1, 1), dep_path.view(-1, 1))
        loss += self.opt['dep_path_loss'] * selection_loss

        mi_label = Variable(
            torch.cat([torch.tensor(np.ones(positive.shape)).float(), torch.tensor(np.zeros(negative.shape)).float()],
                      dim=0)).cuda()
        mi_loss = self.bc(torch.cat([positive, negative], dim=0), mi_label)
        loss += self.opt['mi_loss'] * mi_loss

        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val, sent_loss.item(), selection_loss.item(), mi_loss.item()

    def predict(self, batch, unsort=True):
        inputs, labels, sent_labels, dep_path, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        orig_idx = batch[-1]
        # forward
        self.model.eval()
        logits, sent_logits, _, _, _ = self.model(inputs)

        labels = labels - 1
        labels[labels < 0] = 0
        mask = inputs[1].float()
        mask[mask == 0.] = -1.
        mask[mask == 1.] = 0.
        mask[mask == -1.] = 1.
        mask = mask.byte()
        loss = -self.crf(logits, labels, mask=mask)

        # self.crf.transitions[0][4] = -1
        # self.crf.transitions[0][5] = -1
        # self.crf.transitions[0][6] = -1
        # self.crf.transitions[1][5] = -1
        # self.crf.transitions[1][6] = -1

        probs = F.softmax(logits, dim=1)
        predictions = self.crf.decode(logits, mask=mask)

        sent_predictions = sent_logits.round().long().data.cpu().numpy()

        if unsort:
            _, predictions, probs, sent_predictions = [list(t) for t in zip(
                *sorted(zip(orig_idx, predictions, probs, sent_predictions)))]
        return predictions, probs, loss.item(), sent_predictions
