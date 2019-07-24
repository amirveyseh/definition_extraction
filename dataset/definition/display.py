import spacy
from spacy import displacy

nlp = spacy.load("en")

sentence = ["BOS", "i", "would", "add", "more", "focus", "to", "the", "front", "area", "and", "make", "the", "background", "more", "blurry", "where", "the", "girl", "is", "at", "EOS"]

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
