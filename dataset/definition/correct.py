import json
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
import random

# with open("merged/train.json") as file:
#     dataset = json.load(file)

# for data in dataset:
#     if any([l != 'O' for l in data['labels']]):
#         data['label'] = 'definition'
#     else:
#         data['label'] = 'none'
#
# with open("test.json", 'w') as file:
#     json.dump(dataset, file)


######################################################


# tags = {}
# i = 0
#
# for d in dataset:
#     for l in d['labels']:
#         if l not in tags:
#             i += 1
#             tags[l] = i
#
# print(tags)


#######################################################

# for data in dataset:
#     for i, l in enumerate(data['labels']):
#         if l == 'Alias-Term-frag':
#             data['labels'][i] = 'I-Alias-Term-frag'
#
# with open("dev.json", 'w') as file:
#     json.dump(dataset, file)


#######################################################

# with open('merged2-clipped/dev.json') as file:
#     dataset = json.load(file)
#
# tags = defaultdict(int)
#
# for d in dataset:
#     for l in d['labels']:
#         tags[l] += 1
#
# for k,v in tags.items():
#     print(k,' ',v)


#######################################################

# dataset = []
#
# with open('train.json') as file:
#     dataset += json.load(file)
#
# with open('dev.json') as file:
#     dataset += json.load(file)
#
# with open('test.json') as file:
#     dataset += json.load(file)
#
# lengths = []
#
# for d in dataset:
#     if 'Alias-Term-frag' in d['labels']:
#         dataset.remove(d)
#     if len(d['tokens']) < 200:
#         lengths.append(len(d['tokens']))
#
# lengths = np.asarray(lengths)
#
# print(np.mean(lengths))
# print(np.max(lengths))
# print(len(lengths))
# print(len(dataset))
#
# tags = defaultdict(int)
#
# for d in dataset:
#     for l in d['labels']:
#         tags[l] += 1
#
# for k, v in tags.items():
#     print(k,v)
#
# maps = {
#     'B-Term': 'B-Term',
#     'I-Term': 'I-Term',
#     'B-Definition': 'B-Definition',
#     'I-Definition': 'I-Definition',
#     'B-Definition-frag': 'B-Ordered-Definition',
#     'I-Definition-frag': 'I-Ordered-Definition',
#     'B-Alias-Term': 'B-Alias-Term',
#     'I-Alias-Term': 'I-Alias-Term',
#     'B-Qualifier': 'B-Qualifier',
#     'I-Qualifier': 'I-Qualifier',
#     'B-Definiti-frag': 'B-Ordered-Definition',
#     'I-Definiti-frag': 'I-Ordered-Definition',
#     'B-Secondary-Definition': 'B-Secondary-Definition',
#     'I-Secondary-Definition': 'I-Secondary-Definition',
#     'B-Te-frag': 'B-Ordered-Term',
#     'I-Te-frag': 'I-Ordered-Term',
#     'B-Term-frag': 'B-Ordered-Term',
#     'B-Referential-Definition': 'B-Referential-Definition',
#     'I-Referential-Definition': 'I-Referential-Definition',
#     'B-Referential-Term': 'B-Referential-Term',
#     'I-Referential-Term': 'I-Referential-Term',
#     'B-Secondary-Definiti-frag': 'B-Secondary-Definition',
#     'I-Secondary-Definiti-frag': 'I-Secondary-Definition',
#     'B-Ordered-Term': 'B-Ordered-Term',
#     'I-Ordered-Term': 'I-Ordered-Term',
#     'B-Ordered-Definition': 'B-Ordered-Definition',
#     'I-Ordered-Definition': 'I-Ordered-Definition',
#     'B-Secondary-Definition-frag': 'B-Secondary-Definition',
#     'I-Secondary-Definition-frag': 'I-Secondary-Definition',
#     'B-Alias-Term-frag': 'B-Alias-Term',
#     'I-Alias-Term-frag': 'I-Alias-Term',
#     'B-Qualifier-frag': 'B-Qualifier',
#     'B-Ordered-Definition-frag': 'B-Ordered-Definition',
#     'I-Ordered-Definition-frag': 'I-Ordered-Definition',
#     'I-Qualifier-frag': 'I-Qualifier',
#     'O': 'O'
# }
#
# for k in tags:
#     if k not in maps:
#         print('****: ', k)
#
# assert len(maps) == len(tags), str(len(maps))+' '+str(len(tags))
#
# print(len(set(maps.values())))
# print(set(maps.values()))
#
# newdataset = []
#
# for d in dataset:
#     for i, l in enumerate(d['labels']):
#         d['labels'][i] = maps[l]
#         # pass
#     if len(d['tokens']) < 150:
#         newdataset.append(d)
#
# tags = defaultdict(int)
#
# for d in newdataset:
#     for l in d['labels']:
#         tags[l] += 1
#
# for k, v in tags.items():
#     print(k,v)
#
# def save(dataset, name):
#     train, test, _, _ = train_test_split(dataset, dataset, random_state=1234, train_size=.8)
#     dev = test[:len(test)//2]
#     test = test[len(test)//2:]
#
#     with open(name+'/train.json', 'w') as file:
#         json.dump(train, file)
#     with open(name+'/dev.json', 'w') as file:
#         json.dump(dev, file)
#     with open(name+'/test.json', 'w') as file:
#         json.dump(test, file)
#
# save(dataset, 'merged')
# save(newdataset, 'merged-clipped')
# # save(newdataset, 'clipped')

################################################################################################################

# dataset = []
#
# with open('merged/test.json') as file:
#     dataset += json.load(file)
#
# # tags = defaultdict(int)
# #
# # for d in dataset:
# #     for l in d['labels']:
# #         tags[l] += 1
# #
# # print(tags['B-Term'])
# # print(tags['I-Term'])
# # print(tags['B-Definition'])
# # print(tags['I-Definition'])
# # print(tags['B-Ordered-Term'])
# # print(tags['I-Ordered-Term'])
# # print(tags['B-Ordered-Definition'])
# # print(tags['I-Ordered-Definition'])
# # print(tags['B-Alias-Term'])
# # print(tags['I-Alias-Term'])
# # print(tags['B-Secondary-Definition'])
# # print(tags['I-Secondary-Definition'])
# # print(tags['B-Referential-Term'])
# # print(tags['I-Referential-Term'])
# # print(tags['B-Referential-Definition'])
# # print(tags['I-Referential-Definition'])
# # print(tags['B-Qualifier'])
# # print(tags['I-Qualifier'])
#
# print(len(dataset))


###############################################################################################################


# # f1s = [20, 22, 36, 47, 20, 15, 20, 15, 0, 0, 0, 0, 67, 72, 0, 0, 0, 0]
# f1s = [20, 22, 36, 47, 20, 15, 20, 15, 0, 0, 0, 0, 67, 72]
# counts = [451, 2377, 2131, 87442, 281, 6263, 41, 244, 1, 1, 76, 2096, 2565, 6174]
#
# print(sum(f1s) / len(f1s))
# for i, c in enumerate(counts):
#     f1s[i] *= c
# print(sum(f1s)/sum(counts))
#


########################################################3

# with open('merged2-clipped/dev.json') as file:
#     dataset = json.load(file)
#
# ss = []
#
# for d in dataset:
#     if 'I-Secondary-Definition' in d['labels']:
#         ss.append(d)
#
# ss = random.sample(ss,9)
#
# for s in ss:
#     print(s['tokens'])
#     print(s['labels'])
#     print('++++++++++++++++++++++++++')


##############################################################

# with open('merged2-clipped/train.json') as file:
#     dataset = json.load(file)
#
# multi = 0
# for d in dataset:
#     count = Counter([l for l in map(lambda l: l[1:],d['labels']) if l != ''])
#     if len(count) == 4:
#         # print(d['labels'])
#         # print(count)
#         # exit(1)
#         multi += 1
#
# print(multi/len([d for d in dataset if d['label'] != 'none']))

############################################################################

# labels = ['B-Definition',
# 'I-Definition',
# 'B-Term',
# 'I-Term',
# 'B-Definition-frag',
# 'B-Definiti-frag',
# 'B-Ordered-Definition',
# 'B-Ordered-Definition-frag',
# 'I-Definition-frag',
# 'I-Definiti-frag',
# 'I-Ordered-Definition',
# 'I-Ordered-Definition-frag',
# 'B-Alias-Term',
# 'B-Alias-Term-frag',
# 'I-Alias-Term',
# 'I-Alias-Term-frag',
# 'B-Qualifier',
# 'B-Qualifier-frag',
# 'I-Qualifier',
# 'I-Qualifier-frag',
# 'B-Secondary-Definition',
# 'B-Secondary-Definiti-frag',
# 'B-Secondary-Definition-frag',
# 'I-Secondary-Definition',
# 'I-Secondary-Definiti-frag',
# 'I-Secondary-Definition-frag',
# 'B-Te-frag',
# 'B-Term-frag',
# 'B-Ordered-Term',
# 'I-Te-frag',
# 'I-Ordered-Term',
# 'B-Referential-Definition',
# 'I-Referential-Definition',
# 'B-Referential-Term',
# 'I-Referential-Term']
#
# dataset = []
#
# with open('clipped/train.json') as file:
#     dataset += json.load(file)
# with open('clipped/dev.json') as file:
#     dataset += json.load(file)
# with open('clipped/test.json') as file:
#     dataset += json.load(file)
#
# labels_sent = defaultdict(int)
# for d in dataset:
#     this_label = set()
#     for l in d['labels']:
#         if l == 'O':
#             continue
#         # l = l[2:]
#         if l not in this_label:
#             labels_sent[l] += 1
#             this_label.add(l)
#
# # for k, v in labels.items():
# #     # print(k,v)
# #     print(v)
#
# for l in labels:
#     print(labels_sent[l])


#########################################################

# dataset = []
#
# with open('merged-clipped/train.json') as file:
#     dataset += json.load(file)
# with open('merged-clipped/dev.json') as file:
#     dataset += json.load(file)
# with open('merged-clipped/test.json') as file:
#     dataset += json.load(file)
#
# ss = []
#
# for d in dataset:
#     for l in d['labels']:
#         if l == 'B-Ordered-Term':
#             ss.append(d)
#             continue
#
# for s in ss[:10]:
#     print(s['tokens'])
#     print(s['labels'])
#     print(list(zip(s['tokens'], s['labels'])))
#     print('*********************************************')

###############################################################################

# labels = ['B-Definition',
# 'I-Definition',
# 'B-Term',
# 'I-Term',
# 'B-Definition-frag',
# 'B-Definiti-frag',
# 'B-Ordered-Definition',
# 'B-Ordered-Definition-frag',
# 'I-Definition-frag',
# 'I-Definiti-frag',
# 'I-Ordered-Definition',
# 'I-Ordered-Definition-frag',
# 'B-Alias-Term',
# 'B-Alias-Term-frag',
# 'I-Alias-Term',
# 'I-Alias-Term-frag',
# 'B-Qualifier',
# 'B-Qualifier-frag',
# 'I-Qualifier',
# 'I-Qualifier-frag',
# 'B-Secondary-Definition',
# 'B-Secondary-Definiti-frag',
# 'B-Secondary-Definition-frag',
# 'I-Secondary-Definition',
# 'I-Secondary-Definiti-frag',
# 'I-Secondary-Definition-frag',
# 'B-Te-frag',
# 'B-Term-frag',
# 'B-Ordered-Term',
# 'I-Te-frag',
# 'I-Ordered-Term',
# 'B-Referential-Definition',
# 'I-Referential-Definition',
# 'B-Referential-Term',
# 'I-Referential-Term']
#
# dataset = []
#
# with open('clipped/train.json') as file:
#     dataset += json.load(file)
# with open('clipped/dev.json') as file:
#     dataset += json.load(file)
# with open('clipped/test.json') as file:
#     dataset += json.load(file)
#
# tags = defaultdict(int)
#
# for d in dataset:
#     for l in d['labels']:
#         tags[l] += 1
#
# # for k, v in tags.items():
# #     # print(k,v)
# #     print(v)
#
# for l in labels:
#     print(tags[l])

####################################################################################

# reverse = defaultdict(list)
# for k, v in maps.items():
#     reverse[v].append(k)
#
# for k, v in reverse.items():
#     print(k,v)

##################################################################################

# maps = {
#     'B-Definition': 'B-Definition',
#     'I-Definition': 'I-Definition',
#     'B-Term': 'B-Term',
#     'I-Term': 'I-Term',
#     'B-Ordered-Definition': 'O',
#     'I-Ordered-Definition': 'O',
#     'B-Alias-Term': 'O',
#     'I-Alias-Term': 'O',
#     'B-Qualifier': 'O',
#     'I-Qualifier': 'O',
#     'B-Secondary-Definition': 'O',
#     'I-Secondary-Definition': 'O',
#     'B-Ordered-Term': 'O',
#     'I-Ordered-Term': 'O',
#     'B-Referential-Definition': 'O',
#     'I-Referential-Definition': 'O',
#     'B-Referential-Term': 'O',
#     'I-Referential-Term': 'O',
#     'O': 'O'
# }
#
#
# with open('merged-clipped/train.json') as file:
#     dataset = json.load(file)
#
#
# for d in dataset:
#     for i, l in enumerate(d['labels']):
#         d['labels'][i] = maps[l]
#
# with open('merged-clipped-7/train.json', 'w') as file:
#     json.dump(dataset, file)

#########################################################################################

# dataset = []
#
# with open('merged-clipped-final/train.json') as file:
#     dataset += json.load(file)
# with open('merged-clipped-final/dev.json') as file:
#     dataset += json.load(file)
# with open('merged-clipped-final/test.json') as file:
#     dataset += json.load(file)
#
# terms = set()
# defs = set()
# for i, d in enumerate(dataset):
#     for l in d['labels']:
#         if 'B-Term' in d['labels']:
#             terms.add(i)
#         if 'B-Definition' in d['labels']:
#             defs.add(i)
#
# print(len(terms))
# print(len(defs))
#
# both = set()
#
# for t in terms:
#     if t in defs:
#         both.add(t)
#
# print(len(both))

#########################################################################################
#
# dataset = []
#
# with open('merged-clipped-final/test.json') as file:
#     dataset += json.load(file)
#
# tags = defaultdict(int)
#
# for d in dataset:
#     for l in d['labels']:
#         tags[l] += 1
#
# print(tags)


#######################################################################################

dataset = []

with open('merged-clipped-final/test.json') as file:
    dataset += json.load(file)
with open('merged-clipped-final/train.json') as file:
    dataset += json.load(file)
with open('merged-clipped-final/dev.json') as file:
    dataset += json.load(file)

tags = defaultdict(int)

for d in dataset:
    this_tags = set()
    for l in d['labels']:
        if l != 'O':
            this_tags.add(l)

    for t in this_tags:
        tags[t] += 1

print(tags)