import json
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

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

# with open('train.json') as file:
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
#     'B-Definition-frag': 'B-Ordered-Term',
#     'I-Definition-frag': 'I-Ordered-Term',
#     'B-Alias-Term': 'B-Alias-Term',
#     'I-Alias-Term': 'I-Alias-Term',
#     'B-Qualifier': 'B-Qualifier',
#     'I-Qualifier': 'I-Qualifier',
#     'B-Definiti-frag': 'B-Ordered-Definition',
#     'I-Definiti-frag': 'B-Ordered-Definition',
#     'B-Secondary-Definition': 'B-Secondary-Definition',
#     'I-Secondary-Definition': 'I-Secondary-Definition',
#     'B-Te-frag': 'B-Ordered-Term',
#     'I-Te-frag': 'B-Ordered-Term',
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
#     'I-Secondary-Definition-frag': 'B-Secondary-Definition',
#     'B-Alias-Term-frag': 'B-Alias-Term',
#     'I-Alias-Term-frag': 'I-Alias-Term',
#     'B-Qualifier-frag': 'B-Qualifier',
#     'B-Ordered-Definition-frag': 'B-Ordered-Definition',
#     'I-Ordered-Definition-frag': 'I-Ordered-Definition',
#     'I-Qualifier-frag': 'I-Qualifier',
#     'O': 'O'
# }
#
# assert len(maps) == len(tags)
#
# print(len(set(maps.values())))
# print(set(maps.values()))
#
# newdataset = []
#
# for d in dataset:
#     for i, l in enumerate(d['labels']):
#         d['labels'][i] = maps[l]
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

################################################################################################################

dataset = []

with open('merged/test.json') as file:
    dataset += json.load(file)

# tags = defaultdict(int)
#
# for d in dataset:
#     for l in d['labels']:
#         tags[l] += 1
#
# print(tags['B-Term'])
# print(tags['I-Term'])
# print(tags['B-Definition'])
# print(tags['I-Definition'])
# print(tags['B-Ordered-Term'])
# print(tags['I-Ordered-Term'])
# print(tags['B-Ordered-Definition'])
# print(tags['I-Ordered-Definition'])
# print(tags['B-Alias-Term'])
# print(tags['I-Alias-Term'])
# print(tags['B-Secondary-Definition'])
# print(tags['I-Secondary-Definition'])
# print(tags['B-Referential-Term'])
# print(tags['I-Referential-Term'])
# print(tags['B-Referential-Definition'])
# print(tags['I-Referential-Definition'])
# print(tags['B-Qualifier'])
# print(tags['I-Qualifier'])

print(len(dataset))


