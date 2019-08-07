import json
from collections import Counter
import spacy

nlp = spacy.load("en")

with open('dataset2.json') as file:
    dataset = json.load(file)

terms = []

for d in dataset:
    term = ''
    counter = Counter(d['labels'])
    if counter['B-Term'] == 1 and counter['B-Definition'] == 1:
        for i, l in enumerate(d['labels']):
            if 'Term' in l:
                term += d['tokens'][i]+" "
    if len(term) > 0:
        final_term = ''
        term = nlp(term)
        for token in term:
            if not token.is_stop:
                final_term += token.text + ' '
        if len(final_term) > 0:
            terms.append(final_term)

for term in terms:
    print(term)