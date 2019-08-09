import json
from collections import Counter
import spacy
from tqdm import tqdm

from spacy_wordnet.wordnet_annotator import WordnetAnnotator

nlp = spacy.load("en")
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')

with open("test.json") as file:
    dataset = json.load(file)

for d in dataset:
    terms = [0] * len(d['tokens'])
    if len(d['definition']) > 0:
        term = ''
        start = 0
        pos = 0
        counter = Counter(d['labels'])
        if counter['B-Term'] == 1:
            for i, l in enumerate(d['labels']):
                if l == 'B-Term':
                    start = i
                if 'Term' in l:
                    term += d['tokens'][i] + ' '
            term = nlp(term)
            valid = False
            for i, token in enumerate(term):
                if not token.is_stop:
                    pos = start + i
                    valid = True
                    break
            if valid:
                terms[pos] = 1
    d['terms'] = terms

with open('test.json', 'w') as file:
    json.dump(dataset, file)