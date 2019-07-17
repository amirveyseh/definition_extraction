import json
from collections import Counter
from tqdm import tqdm
import numpy as np

with open('merged-clipped-final/train.json') as file:
    dataset = json.load(file)

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

def get_path(source, destination, dep_path, debug=False):
    if source.id != destination.id:
        dep_path.append(source)
        get_path(source.parent, destination, dep_path, debug=debug)
    else:
        dep_path.append(destination)
    return dep_path

def get_edges(root, edges):
    for child in root.children:
        edges.append((root.id,child.id))
        get_edges(child, edges)
    return edges


def create_adj(term_root, def_root, length):
    adj = np.zeros((length, length))
    edges = get_edges(term_root, [])
    edges += get_edges(def_root, [])
    for edge in edges:
        if edge[0] != -1:
            adj[edge[0]][edge[1]] = 1
            adj[edge[1]][edge[0]] = 1
    return adj


trees = []
dep_paths = []
new_dataset = []

for d in tqdm(dataset):
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
        dep_path = get_path(term_anscestor, lca, [])+get_path(def_anscestor, lca, [])
        dep_path = list(set([n.id for n in dep_path]))
        assert all(id in range(-1, len(d['heads'])) for id in dep_path)
        adj = create_adj(term_anscestor, def_anscestor, len(d['tokens'])).tolist()
        dep_paths.append((dep_path, d, lca.id, term_anscestor.id, def_anscestor.id))
        d['dep_path'] = dep_path
        d['adj'] = adj
        new_dataset.append(d)
    else:
        d['dep_path'] = []
        adj = np.zeros((len(d['dep_path']),len(d['dep_path']))).tolist()
        d['adj'] = adj
        new_dataset.append(d)

with open('lca/train.json', 'w') as file:
    json.dump(new_dataset, file)



