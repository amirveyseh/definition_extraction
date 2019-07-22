import json

dataset = []

with open('lca/test.json') as file:
    dataset += json.load(file)

with open('lca/test.conll', 'w') as file:
    for d in dataset:
        for i in range(len(d['tokens'])):
            file.write(d['tokens'][i]+' '+d['pos'][i]+' '+str(d['heads'][i])+' '+d['labels'][i]+'\n')
        file.write("\n")