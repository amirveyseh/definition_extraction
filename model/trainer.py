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
from model.discriminator import Discriminator
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
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:4]]
        labels = Variable(batch[4].cuda())
    else:
        print("Error")
        exit(1)
    tokens = batch[0]
    head = batch[3]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.discriminator = Discriminator(opt)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.discr_parameters = [p for p in self.discriminator.parameters() if p.requires_grad]
        self.crf = CRF(self.opt['num_class'], batch_first=True)
        self.adversarial_loss = torch.nn.BCELoss()
        if opt['cuda']:
            self.model.cuda()
            self.discriminator.cuda()
            self.criterion.cuda()
            self.crf.cuda()
            self.adversarial_loss.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        self.discr_optimizer = torch_utils.get_optimizer(opt['optim'], self.discr_parameters, opt['lr'])

    def update(self, batch):
        inputs, labels, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        self.discr_optimizer.zero_grad()

        logits = self.model(inputs)

        #### adversarial
        _, masks, _, _ = inputs  # unpack

        gold = Variable(torch.cuda.FloatTensor(labels.size(0), 1).fill_(1.0), requires_grad=False)
        predicted = Variable(torch.cuda.FloatTensor(labels.size(0), 1).fill_(0.0), requires_grad=False)

        labels_vec = labels.data.cpu().numpy().tolist()
        for i, b in enumerate(labels_vec):
            for j, l in enumerate(b):
                labels_vec[i][j] = [0 if k != l-1 else 1 for k in range(self.opt['num_class'])]
        labels_vec = Variable(torch.from_numpy(np.asarray(labels_vec))).float().cuda()

        logits_vec = F.softmax(logits, dim=2)

        pred_loss = self.adversarial_loss(self.discriminator(logits_vec, masks), gold)

        discr_loss = (self.adversarial_loss(self.discriminator(logits_vec, masks), predicted) + self.adversarial_loss(self.discriminator(labels_vec, masks), gold)) / 2

        discr_loss.backward(retain_graph=True)
        self.discr_optimizer.step()
        #############

        labels = labels - 1
        labels[labels < 0] = 0
        mask = inputs[1].float()
        mask[mask == 0.] = -1.
        mask[mask == 1.] = 0.
        mask[mask == -1.] = 1.
        mask = mask.byte()
        loss = -self.crf(logits, labels, mask=mask)

        loss += self.opt['pred_loss']*pred_loss

        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val, pred_loss.item(), discr_loss.item()

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        orig_idx = batch[-1]
        # forward
        self.model.eval()
        logits = self.model(inputs)

        # logits_vec = F.softmax(logits, dim=2)
        # labels = labels.data.cpu().numpy().tolist()
        # for i in range(len(labels)):
        #     if any(l > 1 for l in labels[i]):
        #         for j in range(len(labels[i])):
        #             if labels[i][j] > 1:
        #                 print(logits_vec[i][j])
        #                 print(labels[i][j])
        #                 exit(1)
        # exit(1)

        labels = labels - 1
        labels[labels < 0] = 0
        mask = inputs[1].float()
        mask[mask == 0.] = -1.
        mask[mask == 1.] = 0.
        mask[mask == -1.] = 1.
        mask = mask.byte()
        loss = -self.crf(logits, labels, mask=mask)

        probs = F.softmax(logits, dim=1)
        predictions = self.crf.decode(logits, mask=mask)

        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs)))]
        return predictions, probs, loss.item()
