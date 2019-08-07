import json
from collections import Counter

with open('dataset2.json') as file:
    dataset = json.load(file)

terms = []

for d in dataset:
    term = ''
    counter = Counter(d['labels'])
    if counter['B-Term'] == '1' and counter['B-Definition'] == 1:
        for i, l in enumerate(d['labels']):
            if 'Term' in l:
                term += d['tokens'][i]+" "

print(terms)