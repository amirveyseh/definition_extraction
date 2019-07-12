import spacy
from spacy import displacy

nlp = spacy.load("en")

sentence = ['"', 'VWAP', '"', 'means', ',', 'for', 'any', 'Business', 'Day', ',', 'the', 'volume', '-', 'weighted', 'average', 'price', 'per', 'share', 'of', 'Parent', 'Common', 'Stock', 'on', 'the', 'NYSE', '(', 'as', 'reported', 'by', 'Bloomberg', 'L.P', '.', 'or', ',', 'if', 'not', 'reported', 'therein', ',', 'in', 'another', 'authoritative', 'source', 'mutually', 'selected', 'by', 'the', 'Company', 'and', 'Parent', ')', '.']

doc = nlp(u' '.join(sentence))
print(len(doc))

parse = []

for i, token in enumerate(doc):
    head = token.head.i
    if i == head:
        print(sentence[i-3:i+3])
        head = -1
    parse.append(head)

print(parse)

displacy.serve(doc, style="dep")

##############
