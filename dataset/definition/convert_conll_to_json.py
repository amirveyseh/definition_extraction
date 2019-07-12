import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import spacy

nlp = spacy.load("en")


# from os import listdir
# from os.path import isfile, join
# files = [f for f in listdir('./conll_fixed-clean') if isfile(join('./conll_fixed-clean', f))]
#
# sentences = []
#
# dataset = []
#
# for f in tqdm(files):
#     tokens = []
#     labels = []
#     pos = []
#     label = 'none'
#     with open('./conll_fixed-clean/'+f) as file:
#         lines = file.readlines()
#         sentence = []
#         for l in lines:
#             if l != '\n':
#                 parts = l.strip().split(" ")
#                 sentence.append(parts[0])
#                 tokens.append(parts[0])
#                 pos.append(parts[-2])
#                 labels.append(parts[-6])
#                 if parts[-6] != 'O':
#                     label = 'definition'
#             elif len(sentence) > 0:
#                 the_sentence = ' '.join(sentence)
#                 if the_sentence not in sentences:
#
#                     # if not ' '.join(sentence).startswith(
#                     #         "( b ) The Purchasers agree to the imprinting , so long as is required by this Section 4.1 ,"):
#                     #     continue
#
#                     if len(sentence) < 100:
#                         continue
#
#                     sentences.append(the_sentence)
#                     doc = nlp(u' '.join(sentence))
#                     parse = []
#                     tokens = []
#
#                     print(len(sentence))
#                     print(len(doc))
#                     print(sentence)
#                     print([token.text for token in doc])
#                     exit(1)
#
#                     if len(doc) == len(sentence):
#                         for token in doc:
#                             tokens.append(token.text)
#                             head = token.head.text
#                             if head == token.text:
#                                 ind = -1
#                             else:
#                                 ind = sentence.index(head)
#                             parse.append(ind)
#                         d = {
#                             'tokens': tokens,
#                             'labels': labels,
#                             'pos': pos,
#                             'label': label,
#                             'heads': parse
#                         }
#                         dataset.append(d)
#                 sentence = []
#                 tokens = []
#                 labels = []
#                 pos = []
#                 label = 'none'
#
#
# train, test, _, _ = train_test_split(dataset, dataset, random_state=1234, train_size=.8)
# # train, dev, _, _ = train_test_split(train, train, random_state=1234, train_size=.8)
# dev = test[:len(test)//2]
# test = test[len(test)//2:]
#
# # with open('train.json', 'w') as file:
# #     json.dump(train, file)
# #
# # with open('dev.json', 'w') as file:
# #     json.dump(dev, file)
# #
# # with open('test.json', 'w') as file:
# #     json.dump(test, file)

########################################################################################################################


# with open('test.json') as file:
#     dataset = json.load(file)
#
# print(len(dataset))

########################################################################################################################


#######################################################################################
# with open("./merged-clipped-final/test.json") as file:
#     dataset = json.load(file)
#
# for d in dataset:
#     if ' '.join(d['tokens']).startswith("( b ) The Purchasers agree to the imprinting , so long as is required by this Section 4.1 ,"):
#         print(d)
########################################################################################



##############################################################################
# with open('merged-clipped-final/dev.json') as file:
#     dataset = json.load(file)
#
# for d in tqdm(dataset):
#     doc = nlp(u' '.join(d['tokens']))
#     parse = []
#     for token in doc:
#         head = token.head.i
#         parse.append(head)
#     d['heads'] = parse
#
# with open('merged-clipped-final/dev.json', 'w') as file:
#     json.dump(dataset, file)
##############################################################################


##############################################################################
with open('merged-clipped-final/test.json') as file:
    dataset = json.load(file)

for d in tqdm(dataset):
    for i in range(len(d['heads'])):
        if d['heads'][i] == i:
            d['heads'][i] = -1

with open('merged-clipped-final/test.json', 'w') as file:
    json.dump(dataset, file)
##############################################################################