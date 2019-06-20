"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        mask = inputs[1]
        negatives = labels.eq(1).sum().item()
        positives = (labels.eq(1).eq(0).sum() - mask.sum()).item()
        ratio = positives / negatives
        mask_o = torch.Tensor(np.random.rand(*labels.shape)).cuda()
        mask_o[mask_o < ratio] = 0.0
        mask_o[mask_o >= ratio] = 1.0
        mask_o = mask_o.byte() * labels.eq(1)
        mask = mask.masked_fill(mask_o, 1.0)

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(inputs)

        labels = labels - 1
        labels[labels < 0] = 0
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        mask = mask.view(-1)
        # final_loss = 0
        # c = 0
        # for i in range(len(loss)):
        #     if mask[i] != 1.0:
        #         final_loss += loss[i]
        #         c += 1
        # print(c)
        # exit(1)
        # loss = final_loss / c

        loss = loss.masked_fill(mask, 0).mean()

        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        mask = inputs[1]

        orig_idx = batch[5]
        # forward
        self.model.eval()
        logits = self.model(inputs)

        labels = labels - 1
        labels[labels < 0] = 0
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        mask = mask.view(-1)
        # final_loss = 0
        # c = 0
        # for i in range(len(loss)):
        #     if mask[i] != 1.0:
        #         final_loss += loss[i]
        #         c += 1
        # loss = final_loss / c

        loss = loss.masked_fill(mask, 0).mean()

        probs = F.softmax(logits.view(-1, logits.shape[-1]), 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.view(-1, logits.shape[-1]).data.cpu().numpy(), axis=1).tolist()

        mask = mask.view(logits.shape[0], -1)

        final_predictions = []
        final_probs = []
        for i in range(mask.shape[0]):
            pred = []
            prob = []
            for j in range(mask.shape[1]):
                if mask[i][j] != 1.0:
                    pred.append(predictions[i])
                    prob.append(probs[i])
            final_predictions.append(pred)
            final_probs.append(prob)

        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx, final_predictions, final_probs)))]
        return predictions, probs, loss.item()
