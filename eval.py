"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--per_class', type=int, default=0, help="")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v, k) for k, v in label2id.items()])

predictions = []
all_probs = []
words = []
batch_iter = tqdm(batch)
for i, b in enumerate(batch_iter):
    preds, probs, _, word = trainer.predict(b)
    predictions += preds
    all_probs += probs
    words += [vocab.unmap([id for id in w if id != constant.PAD_ID]) for w in word]

lens = [len(word) for word in words]

########################################

def repack(tokens, lens):
    output = []
    token = []
    i = 0
    j = 0
    for t in tokens:
        t = t[0]
        if j < lens[i]:
            token.append(t)
        else:
            j = 0
            output.append(token)
            token = []
            token.append(t)
            i += 1
        j += 1
    if len(token) > 0:
        output.append(token)
    return output

predictions_ = [[id2label[l+1] for l in p] for p in predictions]
gold = repack(batch.gold(), lens)
corrections = []

incorrect = 0
bad = 0
sent_bads = []
for i, p in enumerate(predictions_):
    check_incorrect = False
    if any([p[j] != gold[i][j] for j in range(len(p))]):
        incorrect += 1
        check_incorrect = True
    check_bad = False
    for k in range(len(p)-1):
        if p[k+1] == 'I-Definition' or p[k+1] == 'I-Term' or p[k+1] == 'I-Qualifier':
            if p[k][2:] != p[k+1][2:]:
                bad += 1
                check_bad = True
                break
    if check_incorrect and check_bad:
        sent_incorrect = 0
        sent_bad = 0
        for k in range(len(p)):
            if p[k] != gold[i][k]:
                sent_incorrect += 1
        for k in range(len(p)-1):
            if p[k + 1] == 'I-Definition' or p[k + 1] == 'I-Term' or p[k + 1] == 'I-Qualifier':
                if p[k][2:] != p[k + 1][2:]:
                    sent_bad += 1
                    corrections.append((p[k], p[k+1], gold[i][k], gold[i][k+1]))
                    predictions[i][k+1] = label2id[gold[i][k+1]]-1
                    # predictions[i][k+1] = predictions[i][k]
        sent_bads.append(sent_bad/sent_incorrect)
print(bad/incorrect)
print(sum(sent_bads)/len(sent_bads))
print(corrections)
########################################

predictions = [[id2label[l + 1]] for p in predictions for l in p]
words = [[w] for word in words for w in word]
print(len(predictions))
print(len(batch.gold()))
print(len(words))
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True, verbose_output=args.per_class == 1)

print('scroes from sklearn: ')
macro_f1 = f1_score(batch.gold(), predictions, average='macro')
micro_f1 = f1_score(batch.gold(), predictions, average='micro')
macro_p = precision_score(batch.gold(), predictions, average='macro')
micro_p = precision_score(batch.gold(), predictions, average='micro')
macro_r = recall_score(batch.gold(), predictions, average='macro')
micro_r = recall_score(batch.gold(), predictions, average='micro')
print('micro scores: ')
print('micro P: ', micro_p)
print('micro R: ', micro_r)
print('micro F1: ', micro_f1)
print("")
print("macro scroes: ")
print('macro P: ', macro_p)
print('macro R: ', macro_r)
print('macro F1: ', macro_f1)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset, p, r, f1))

cm = confusion_matrix(batch.gold(), predictions, labels=['B-Term', 'I-Term', 'B-Definition', 'I-Definition',
                                                         'B-Qualifier', 'I-Qualifier', 'O'])
with open('report/confusion_matrix.txt', 'w') as file:
    for row in cm:
        file.write(('{:5d},' * len(row)).format(*row.tolist())+'\n')
print("confusion matrix created!")

def repack(tokens, lens):
    output = []
    token = []
    i = 0
    j = 0
    for t in tokens:
        t = t[0]
        if j < lens[i]:
            token.append(t)
        else:
            j = 0
            output.append(token)
            token = []
            token.append(t)
            i += 1
        j += 1
    return output

predictions = repack(predictions, lens)
gold = repack(batch.gold(), lens)
words = repack(words, lens)

########################################
# incorrect = 0
# correct = 0
#
# for p in predictions:
#     for i in range(len(p)-1):
#         if p[i+1] == 'I-Definition' or p[i+1] == 'I-Term' or p[i+1] == 'I-Qualifier':
#             if p[i][2:] != p[i+1][2:]:
#                 incorrect += 1
#             else:
#                 correct += 1
# print(incorrect)
########################################


########################################
# ss = []
# for i, p in enumerate(predictions):
#     for j, l in enumerate(p):
#         if l == 'I-Qualifier' and gold[i][j] == 'I-Definition':
#             ss.append({
#                 'words': words[i],
#                 'gold': gold[i],
#                 'pred': predictions[i]
#             })
#
# print(len(ss))
# print(ss[0])
########################################


########################################
# false_positive = 0
# false_negative = 0
# true_positive = 0
# true_negative = 0
#
# for i, p in enumerate(predictions):
#     if any([l != 'O' for l in gold[i]]):
#         if any([l != 'O' for l in p]):
#             true_positive += 1
#         elif all([l == 'O' for l in p]):
#             false_negative += 1
#     elif all([l == 'O' for l in gold[i]]):
#         if any([l != 'O' for l in p]):
#             false_positive += 1
#         elif all([l == 'O' for l in p]):
#             true_negative += 1
#
# print(true_negative/(true_negative+false_positive))
########################################


########################################
# incorrect = 0
# bad = 0
# sent_bads = []
# for i, p in enumerate(predictions):
#     check_incorrect = False
#     if any([p[j] != gold[i][j] for j in range(len(p))]):
#         incorrect += 1
#         check_incorrect = True
#     check_bad = False
#     for k in range(len(p)-1):
#         if p[k+1] == 'I-Definition' or p[k+1] == 'I-Term' or p[k+1] == 'I-Qualifier':
#             if p[k][2:] != p[k+1][2:]:
#                 bad += 1
#                 check_bad = True
#                 break
#     if check_incorrect and check_bad:
#         sent_incorrect = 0
#         sent_bad = 0
#         for k in range(len(p)):
#             if p[k] != gold[i][k]:
#                 sent_incorrect += 1
#         for k in range(len(p)-1):
#             if p[k + 1] == 'I-Definition' or p[k + 1] == 'I-Term' or p[k + 1] == 'I-Qualifier':
#                 if p[k][2:] != p[k + 1][2:]:
#                     sent_bad += 1
#         sent_bads.append(sent_bad/sent_incorrect)
# print(bad/incorrect)
# print(sum(sent_bads)/len(sent_bads))
########################################

print("Evaluation ended.")
