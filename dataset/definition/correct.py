import json

with open("dev.json") as file:
    dataset = json.load(file)

# for data in dataset:
#     if any([l != 'O' for l in data['labels']]):
#         data['label'] = 'definition'
#     else:
#         data['label'] = 'none'
#
# with open("test.json", 'w') as file:
#     json.dump(dataset, file)


######################################################


# tags = {}
# i = 0
#
# for d in dataset:
#     for l in d['labels']:
#         if l not in tags:
#             i += 1
#             tags[l] = i
#
# print(tags)


#######################################################

# for data in dataset:
#     for i, l in enumerate(data['labels']):
#         if l == 'Alias-Term-frag':
#             data['labels'][i] = 'I-Alias-Term-frag'
#
# with open("dev.json", 'w') as file:
#     json.dump(dataset, file)


#######################################################



