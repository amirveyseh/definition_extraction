# import json
# from collections import Counter
# import spacy
#
# nlp = spacy.load("en")
#
# with open('dataset2.json') as file:
#     dataset = json.load(file)
#
# terms = []
#
# for d in dataset:
#     term = ''
#     counter = Counter(d['labels'])
#     if counter['B-Term'] == 1 and counter['B-Definition'] == 1:
#         for i, l in enumerate(d['labels']):
#             if 'Term' in l:
#                 term += d['tokens'][i]+" "
#     if len(term) > 0:
#         final_term = ''
#         term = nlp(term)
#         for token in term:
#             if not token.is_stop:
#                 final_term += token.text + ' '
#         if len(final_term) > 0:
#             terms.append(final_term)
#
# for term in terms:
#     print(term)


#######################################################################################################

import json
from colorama import Fore, Back, Style

def get_colored_text(t_l):
    text = ""
    for t in t_l:
        if 'Term' in t[1]:
            color = Fore.GREEN
        elif 'Definition' in t[1]:
            color = Fore.RED
        elif 'Qualifier' in t[1]:
            color = Fore.YELLOW
        else:
            color = Fore.WHITE
        text += color + t[0] + ' '
    return text

def get_colored_text_html(t_l):
    text = ""
    for t in t_l:
        if 'Term' in t[1]:
            text += "<font color='green'>"+t[0] + ' ' + "</font>"
        elif 'Definition' in t[1]:
            text += "<font color='red'>" + t[0] + ' ' + "</font>"
        elif 'Qualifier' in t[1]:
            text += "<font color='blue'>" + t[0] + ' ' + "</font>"
        else:
            text += "<font color='black'>" + t[0] + ' ' + "</font>"
    text += "<br/><br/>"
    return text

with open('analysis/mis_labeled.json') as file:
    dataset = json.load(file)

# for d in dataset[:50]:
#     print("Ground Truth: ")
#     print(get_colored_text(d[0]))
#     print(Style.RESET_ALL)
#     print("=================================================")
#     print("Prediction: ")
#     print(get_colored_text(d[2]))
#     print(Style.RESET_ALL)
#     print("=========================================================================")
#     print("||                                                                      ||")
#     print("||                                                                      ||")
#     print("||                                                                      ||")
#     print("=========================================================================")

with open("mis_labeled.html", 'w') as file:
    for d in dataset:
        file.write("Ground Truth: <br/>")
        file.write(get_colored_text_html(d[0]))
        file.write("---------------------------------------------------------------------------------------------<br/>")
        file.write("Prediction: <br/>")
        file.write(get_colored_text_html(d[2]))
        file.write("=========================================================================<br/>")
        file.write("=========================================================================<br/>")
        file.write("=========================================================================<br/>")