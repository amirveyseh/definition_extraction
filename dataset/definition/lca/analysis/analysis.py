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

with open('mis_labeled.json') as file:
    dataset = json.load(file)

for d in dataset[:50]:
    print("Gold: ")
    print(get_colored_text(d[0]))
    print(Style.RESET_ALL)
    print("=================================================")
    print("Prediction: ")
    print(get_colored_text(d[2]))
    print(Style.RESET_ALL)
    print("=========================================================================")
    print("||                                                                      ||")
    print("||                                                                      ||")
    print("||                                                                      ||")
    print("=========================================================================")