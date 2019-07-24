import json
# import spacy
# from spacy.tokens import Doc
#
# nlp = spacy.load("en")
#
# dataset = []
#
# with open("test.tsv") as file:
#     lines = file.readlines()
#     for l in lines:
#         parts = l.strip().split('\t')
#         labels = parts[1].split(',')
#         sentence = parts[2]
#
#         label_parsed = []
#
#         for label in labels:
#             label_parts = label.split(":")
#             label_parsed.append([int(label_parts[0]), int(label_parts[1]), label_parts[2], 0, 0])
#
#         for i in range(len(sentence)):
#             c = sentence[i]
#             if c == ' ':
#                 for j, label in enumerate(label_parsed):
#                     if label[0] > i:
#                         label_parsed[j][3] += 1
#                     if label[1] > i:
#                         label_parsed[j][4] += 1
#
#         for label in label_parsed:
#             label[0] -= label[3]
#             label[1] -= label[4]
#
#         doc = nlp(sentence)
#         parse = []
#         tokens = []
#         pos = []
#         for i, token in enumerate(doc):
#             head = token.head.i
#             tokens.append(token.text)
#             pos.append(token.pos_)
#             if i == head:
#                 head = -1
#             parse.append(head)
#
#         t = 0
#         for token in tokens:
#             t += len(token)
#         assert t == len(sentence.replace(' ', ''))
#
#         tags = []
#         i = 0
#         for token in tokens:
#             tag = ''
#             for label in label_parsed:
#                 if i == label[0]:
#                     tag = 'B-' + label[2][2:]
#                 elif label[0] < i < label[1]:
#                     tag = 'I-' + label[2][2:]
#             if tag == '':
#                 tag = 'O'
#             tags.append(tag)
#             i += len(token)
#
#
#         assert len(tokens) == len(tags) == len(parse) == len(pos)
#         for i in range(1, len(tags)):
#             if tags[i].startswith('I') and tags[i][2:] != tags[i-1][2:]:
#                 print('error')
#                 exit(1)
#         tag_set = set([t[2:] for t in tags if t != 'O'])
#         orig_tag_set = set([l[2][2:] for l in label_parsed])
#         assert tag_set == orig_tag_set
#
#         dataset.append({
#             'tokens': tokens,
#             'labels': tags,
#             'head': parse,
#             'pos': pos
#         })
#
# with open('test.json', 'w') as file:
#     json.dump(dataset, file)
#

#################################################################################################

dataset = []

with open('train.json') as file:
    dataset += json.load(file)
with open('train.json') as file:
    dataset += json.load(file)
with open('train.json') as file:
    dataset += json.load(file)

label_set = set()

for d in dataset:
    for label in d['labels']:
        label_set.add(label)

i = 1
label_ = {}
for label in label_set:
    label_[label] = i
    i += 1
print(label_)