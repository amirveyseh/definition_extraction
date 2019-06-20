import json

with open("test.json") as file:
    dataset = json.load(file)

for data in dataset:
    if any([l != 'O' for l in data['labels']]):
        data['label'] = 'definition'
    else:
        data['label'] = 'none'

with open("test.json", 'w') as file:
    json.dump(dataset, file)


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

