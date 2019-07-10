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

sent_label2id = constant.SENT_LABEL_TO_ID
sent_id2label = dict([(v, k) for k, v in sent_label2id.items()])

predictions = []
all_probs = []
sent_predictions = []
batch_iter = tqdm(batch)
for i, b in enumerate(batch_iter):
    preds, probs, _, sent_preds = trainer.predict(b)
    predictions += preds
    all_probs += probs
    sent_predictions += sent_preds

predictions = [[id2label[l + 1]] for p in predictions for l in p]
sent_predictions = [sent_id2label[p] for p in sent_predictions]
print(len(predictions))
print(len(batch.gold()))
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
                                                         'B-Ordered-Term', 'I-Ordered-Term', 'B-Ordered-Definition',
                                                         'I-Ordered-Definition', 'B-Alias-Term',
                                                         'I-Alias-Term', 'B-Secondary-Definition',
                                                         'I-Secondary-Definition',
                                                         'B-Qualifier', 'I-Qualifier', 'O'])
with open('report/confusion_matrix.txt', 'w') as file:
    for row in cm:
        file.write(('{:5d},' * len(row)).format(*row.tolist())+'\n')
print("confusion matrix created!")

print('sentence predicitons accuracy: ', sum([1 if sent_predictions[i] == batch.sent_gold()[i] else 0 for i in range(len(sent_predictions))])/len(sent_predictions))

# p, r, f1 = scorer.score(batch.sent_gold(), sent_predictions, verbose=True, verbose_output=args.per_class == 1, task='sent')
# print('sent p: ', p)
# print('sent r: ', r)
# print('sent f1: ', f1)

print("Evaluation ended.")
