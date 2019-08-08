import json
from collections import Counter
import spacy
from tqdm import tqdm

from spacy_wordnet.wordnet_annotator import WordnetAnnotator

nlp = spacy.load("en")
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')

with open('dataset2.json') as file:
    dataset = json.load(file)

terms = []
terms2 = []
tt = 0

for d in dataset:
    term = ''
    definition = []
    counter = Counter(d['labels'])
    if counter['B-Term'] == 1:
        tt += 1
        for i, l in enumerate(d['labels']):
            if 'Term' in l:
                term += d['tokens'][i]+" "
    if len(term) > 0:
        final_term = ''
        term = nlp(term)
        for token in term:
            if not token.is_stop:
                final_term += token.text + ' '
        final_term = final_term.strip()
        if final_term.count(' ') == 0 and len(final_term) > 0:
            terms.append(final_term)

            token = nlp(final_term)[0]
            synset = token._.wordnet.synsets()
            if len(synset) > 0 and len(synset[0].definition()) > 0:
                definition = synset[0].definition().split(' ')
    d['definition'] = definition

with open('dataset3.json', 'w') as file:
    json.dump(dataset, file)

#######################################################################################################

# import json
# from colorama import Fore, Back, Style
#
# def get_colored_text(t_l):
#     text = ""
#     for t in t_l:
#         if 'Term' in t[1]:
#             color = Fore.GREEN
#         elif 'Definition' in t[1]:
#             color = Fore.RED
#         elif 'Qualifier' in t[1]:
#             color = Fore.YELLOW
#         else:
#             color = Fore.WHITE
#         text += color + t[0] + ' '
#     return text
#
# def get_colored_text_html(t_l):
#     text = ""
#     for t in t_l:
#         if 'Term' in t[1]:
#             text += "<font color='green'>"+t[0] + ' ' + "</font>"
#         elif 'Definition' in t[1]:
#             text += "<font color='red'>" + t[0] + ' ' + "</font>"
#         elif 'Qualifier' in t[1]:
#             text += "<font color='blue'>" + t[0] + ' ' + "</font>"
#         else:
#             text += "<font color='black'>" + t[0] + ' ' + "</font>"
#     text += "<br/><br/>"
#     return text
#
# with open('analysis/mis_labeled.json') as file:
#     dataset = json.load(file)
#
# # for d in dataset[:50]:
# #     print("Ground Truth: ")
# #     print(get_colored_text(d[0]))
# #     print(Style.RESET_ALL)
# #     print("=================================================")
# #     print("Prediction: ")
# #     print(get_colored_text(d[2]))
# #     print(Style.RESET_ALL)
# #     print("=========================================================================")
# #     print("||                                                                      ||")
# #     print("||                                                                      ||")
# #     print("||                                                                      ||")
# #     print("=========================================================================")
#
# with open("mis_labeled.html", 'w') as file:
#     for d in dataset:
#         file.write("Ground Truth: <br/>")
#         file.write(get_colored_text_html(d[0]))
#         file.write("---------------------------------------------------------------------------------------------<br/>")
#         file.write("Prediction: <br/>")
#         file.write(get_colored_text_html(d[2]))
#         file.write("=========================================================================<br/>")
#         file.write("=========================================================================<br/>")
#         file.write("=========================================================================<br/>")

###################################################################################################################

# good = 0
# num_syns = []
# for i, term in tqdm(enumerate(terms)):
#     valid = True
#     definition = []
#     for t in term:
#         nlp = spacy.load('en')
#         nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
#         token = nlp(t)[0]
#         synset = token._.wordnet.synsets()
#         num_syns.append(len(synset))
#         if len(synset) == 0 or len(synset[0].definition()) == 0:
#             valid = False
#             break
#     if valid:
#         good += 1
#     if i % 100 == 0 and i > 0:
#         # print(good / i)
#         print(sum(num_syns)/len(num_syns))
#
# print(good / len(terms))