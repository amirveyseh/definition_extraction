import json
from collections import Counter

dataset = []

with open('train.json') as file:
    dataset += json.load(file)
with open('train.json') as file:
    dataset += json.load(file)
with open('train.json') as file:
    dataset += json.load(file)

term_def = []

for d in dataset:
    term = ''
    count = Counter(d['labels'])
    if count['B-Term'] == 1:
        for i, l in enumerate(d['labels']):
            if 'Term' in l:
                term += d['tokens'][i] + ' '
        if len(d['definition']) > 0:
            term_def.append([term, d['definition'], d['tokens']])

for td in term_def:
    print(td)