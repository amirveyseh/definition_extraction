import json

dataset = []

with open("lca/train.json") as file:
    dataset += json.load(file)
with open("lca/dev.json") as file:
    dataset += json.load(file)
with open("lca/test.json") as file:
    dataset += json.load(file)

words = set()

for d in dataset:
    for t in d['tokens']:
        words.add(t)

with open('lca/bert_vocab.txt', 'w') as file:
    for w in words:
        file.write(w+'\n')
