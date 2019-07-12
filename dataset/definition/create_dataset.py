import json
from collections import Counter

class Tree():
    def __init__(self, id):
        self.children = []
        self.parent = None
        self.text = ""
        self.descendants = []
        self.id = id
        self.level = 0

def get_anscestors(node, ids):
    valids = []
    if all(id in node.descendants for id in ids):
        valids.append(node)
        for child in node.children:
            valids.extend(get_anscestors(child, ids))
    return valids

def get_lca(node, ids):
    ancestors = get_anscestors(node, ids)
    if len(ancestors) == 0:
        print(ids)
        print(node.descendants)
        exit(1)
    return sorted(ancestors, key=lambda n: n.level)[-1]


def augment(node, level = 0):
    node.descendants.append(node.id)
    node.level = level
    for child in node.children:
        augment(child, level=level+1)
        node.descendants.extend(child.descendants)

def get_path(source, destination, path = []):
    if source.id != destination.id:
        path.append(source)
        get_path(source.parent, destination)
    else:
        path.append(destination)
    return path




with open('merged-clipped-final/test.json') as file:
    dataset = json.load(file)



trees = []
dep_paths = []
new_dataset = []

for d in dataset:
    nodes = {}
    root = Tree(-1)
    nodes[-1] = root
    for i in range(len(d['tokens'])):
        node = Tree(i)
        node.text = d['tokens'][i]
        nodes[i] = node
    for i in range(len(d['heads'])):
        nodes[i].parent = nodes[d['heads'][i]]
        nodes[d['heads'][i]].children.append(nodes[i])
    trees.append(root)

    augment(root)

    assert len(root.descendants) == len(d['tokens'])+1

    count = Counter(d['labels'])
    if count['B-Definition'] == 1 and count['B-Term'] == 1:
        terms = []
        defs = []
        for i, l in enumerate(d['labels']):
            if l == 'B-Term' or l == 'I-Term':
                terms += [i]
            if l == 'B-Definition' or l == 'I-Definition':
                defs += [i]

        term_anscestor = get_lca(root, terms)
        def_anscestor = get_lca(root, defs)
        lca = get_lca(root, [term_anscestor.id, def_anscestor.id])
        dep_path = get_path(term_anscestor, lca)+get_path(def_anscestor, lca)
        dep_path = list(set([n.id for n in dep_path]))
        dep_paths.append((dep_path, d, lca.id))
        d['lca'] = lca.id
        new_dataset.append(d)
    elif count['B-Definition'] == 0 and count['B-Term'] == 0:
        dep_paths.append((None, d, -2))
        d['lca'] = -2
        new_dataset.append(d)

with open('lca/test.json', 'w') as file:
    json.dump(new_dataset, file)



