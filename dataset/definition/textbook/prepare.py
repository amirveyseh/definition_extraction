import glob, json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import spacy
nlp = spacy.load("en")

# path = 'data/'
#
# files = [f for f in glob.glob(path + "**/*.conll", recursive=True)]
#
# content = ''
#
# for f in files:
#     with open(f) as file:
#         content += file.read()
#
# with open('dataset.conll', 'w') as file:
#     file.write(content)

###############################################################################################

# with open('dataset.conll') as file:
#     lines = file.readlines()
#
# dataset = []
#
# tokens = []
# labels = []
#
# for l in lines:
#     l = l.strip()
#     if len(l) == 0:
#         if len(tokens) > 0:
#             assert len(tokens) == len(labels)
#             dataset.append({
#                 'tokens': tokens,
#                 'labels': labels
#             })
#         tokens = []
#         labels = []
#     else:
#         parts = list(map(lambda a: a.strip(), l.split('\t')))
#         tokens.append(parts[0])
#         labels.append(parts[4])

################################################################################################

# with open('dataset.json') as file:
#     dataset = json.load(file)
#
# label_map = {
#     'B-Alias-Term': 'O',
#     'B-Qualifier': 'B-Qualifier',
#     'I-Referential-Definition': 'O',
#     'Secondary-Definition': 'O',
#     'I-Alias-Term': 'O',
#     'I-Ordered-Definition': 'O',
#     'B-Definition': 'B-Definition',
#     'Term': 'B-Term',
#     'I-Qualifier': 'I-Qualifier',
#     'B-Referential-Term': 'O',
#     'B-Term': 'B-Term',
#     'I-Term': 'I-Term',
#     'I-Definiti-frag': 'O',
#     'Alias-Term': 'O',
#     'B-Referential-Definition': 'O',
#     'I-Definition': 'I-Definition',
#     'B-Definiti-frag': 'O',
#     'I-Secondary-Definition': 'O',
#     'B-Ordered-Term': 'O',
#     'Definition': 'B-Definition',
#     'Referential-Definition': 'O',
#     'I-Ordered-Term': 'O',
#     'Referential-Term': 'O',
#     'I-Referential-Term': 'O',
#     'I-Te-frag': 'O',
#     'Definiti-frag': 'O',
#     'B-Secondary-Definition': 'O',
#     'Qualifier': 'B-Qualifier',
#     'B-Alias-Te-frag': 'O',
#     'B-Te-frag': 'O',
#     'B-Ordered-Definition': 'O',
#     'O': 'O'
# }
#
# for d in dataset:
#     for i, l in enumerate(d['labels']):
#         d['labels'][i] = label_map[l]

#####################################################################################################

# with open('dataset2.json') as file:
#     dataset = json.load(file)

# for d in dataset:
#     for i in range(1,len(d['labels'])):
#         if d['labels'][i].startswith('I') and d['labels'][i][2:] != d['labels'][i-1][2:]:
#             print(d)
#             exit(1)

# for d in dataset:
#     for i in range(1,len(d['labels'])):
#         if d['labels'][i] == 'B-Qualifier' and d['labels'][i-1] == 'B-Qualifier':
#             for k in range(i,len(d['labels'])):
#                 if d['labels'][k] == 'B-Qualifier':
#                     d['labels'][k] = 'I-Qualifier'
#                 else:
#                     break
#
# for d in dataset:
#     for i in range(1,len(d['labels'])):
#         if d['labels'][i] == 'B-Qualifier' and d['labels'][i-1] == 'B-Qualifier':
#             print(d)
#             exit(1)

#######################################################################################################

# with open('dataset2.json') as file:
#     dataset = json.load(file)
#
# newdataset = []
#
# for d in tqdm(dataset):
#     doc = nlp(' '.join(d['tokens']))
#     parse = []
#     tokens = []
#     pos = []
#     for i, token in enumerate(doc):
#         head = token.head.i
#         tokens.append(token.text)
#         pos.append(token.pos_)
#         if i == head:
#             head = -1
#         parse.append(head)
#
#     k = 0
#     m = 0
#     labels = []
#     while k < len(d['tokens']):
#         wl = len(d['tokens'][k])
#         wl2 = len(tokens[m])
#         num = 1
#         while wl2 < wl:
#             m += 1
#             wl2 += len(tokens[m])
#             num += 1
#         assert wl == wl2
#         label = d['labels'][k]
#         labels += [label] * num
#         k += 1
#         m += 1
#
#     assert len(parse) == len(pos) == len(tokens) == len(labels)
#
#     newdataset.append({
#         'tokens': tokens,
#         'labels': labels,
#         'heads': parse,
#         'pos': pos,
#         'label': 'definition' if any(l != 'O' for l in labels) else 'none'
#     })
#
# with open('dataset2.json', 'w') as file:
#     json.dump(newdataset, file)

########################################################################################################

# with open('dataset2.json') as file:
#     dataset = json.load(file)
#
# train, test, _, _ = train_test_split(dataset, dataset, random_state=1234, train_size=.8)
# # train, dev, _, _ = train_test_split(train, train, random_state=1234, train_size=.8)
# dev = test[:len(test)//2]
# test = test[len(test)//2:]
#
# with open('train.json', 'w') as file:
#     json.dump(train, file)
#
# with open('dev.json', 'w') as file:
#     json.dump(dev, file)
#
# with open('test.json', 'w') as file:
#     json.dump(test, file)

###########################################################################################################

with open('dataset2.json') as file:
    dataset = json.load(file)

# maxlen = 0
#
# for d in dataset:
#     if len(d['tokens']) > maxlen:
#         maxlen = len(d['tokens'])
#
# print(maxlen)


print(len(dataset))