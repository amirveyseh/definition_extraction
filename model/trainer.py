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
from model.main import Main
from model.auxilary1 import Auxilary1
from model.auxilary2 import Auxilary2
from model.auxilary3 import Auxilary3
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
        inputs = [Variable(b.cuda()) for b in batch[:7]]
        labels = Variable(batch[7].cuda())
        sent_labels = Variable(batch[8].cuda())
        dep_path = Variable(batch[9].cuda())
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
        self.main = Main(opt)
        self.auxilary1 = Auxilary1(opt)
        self.auxilary2 = Auxilary2(opt)
        self.auxilary3 = Auxilary3(opt)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.main_parameters = [p for p in self.main.parameters() if p.requires_grad]
        self.auxilary1_parameters = [p for p in self.auxilary1.parameters() if p.requires_grad]
        self.auxilary2_parameters = [p for p in self.auxilary2.parameters() if p.requires_grad]
        self.auxilary3_parameters = [p for p in self.auxilary3.parameters() if p.requires_grad]
        self.crf = CRF(self.opt['num_class'], batch_first=True)
        self.bc = nn.BCELoss()
        self.kl = nn.KLDivLoss()
        if opt['cuda']:
            self.model.cuda()
            self.main.cuda()
            self.auxilary1.cuda()
            self.auxilary2.cuda()
            self.auxilary3.cuda()
            self.criterion.cuda()
            self.crf.cuda()
            self.bc.cuda()
            self.kl.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        self.main_optimizer = torch_utils.get_optimizer(opt['optim'], self.main_parameters, opt['lr'])
        self.auxilary1_optimizer = torch_utils.get_optimizer(opt['optim'], self.auxilary1_parameters, opt['lr'])
        self.auxilary2_optimizer = torch_utils.get_optimizer(opt['optim'], self.auxilary2_parameters, opt['lr'])
        self.auxilary3_optimizer = torch_utils.get_optimizer(opt['optim'], self.auxilary3_parameters, opt['lr'])


    def update(self, batch):
        inputs, labels, sent_labels, dep_path, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        _, masks, _, _, terms, defs, _ = inputs

        # step forward
        self.model.train()
        self.main.train()
        self.optimizer.zero_grad()
        self.main_optimizer.zero_grad()
        self.auxilary1_optimizer.zero_grad()
        self.auxilary2_optimizer.zero_grad()
        self.auxilary3_optimizer.zero_grad()
        main_inputs = self.model(inputs)
        logits, class_logits, selections, term_def, not_term_def = self.main(main_inputs, masks, terms, defs)

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

        term_def_loss = -self.opt['consistency_loss'] * (term_def-not_term_def)
        loss += term_def_loss


        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.main.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        self.main_optimizer.step()

        sf = nn.Softmax(2)
        logits1 = sf(self.auxilary1(main_inputs, masks, terms, defs))
        logits2 = sf(self.auxilary2(main_inputs, masks, terms, defs))
        logits3 = sf(self.auxilary3(main_inputs, masks, terms, defs))
        logits = sf(logits)

        loss1 = F.kl(torch.log(logits1+constant.eps), logits)
        loss2 = F.kl(torch.log(logits2+constant.eps), logits)
        loss3 = F.kl(torch.log(logits3+constant.eps), logits)
        # loss1 = F.kl_div(logits, logits3)

        loss = loss1 + loss2 + loss3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.auxilary1.parameters(), self.opt['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.auxilary2.parameters(), self.opt['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.auxilary3.parameters(), self.opt['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        self.auxilary1_optimizer.step()
        self.auxilary2_optimizer.step()
        self.auxilary3_optimizer.step()

        return loss_val, sent_loss.item(), selection_loss.item()

    def predict(self, batch, unsort=True):
        inputs, labels, sent_labels, dep_path, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        _, masks, _, _, terms, defs, _ = inputs

        orig_idx = batch[-1]
        # forward
        self.model.eval()
        self.main.eval()
        main_inputs = self.model(inputs)
        logits, sent_logits, _, _, _ = self.main(main_inputs, masks, terms, defs)

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
            _, predictions, probs, sent_predictions = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs, sent_predictions)))]
        return predictions, probs, loss.item(), sent_predictions
