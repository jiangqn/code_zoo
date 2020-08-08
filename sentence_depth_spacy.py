# compute the depth of dependency tree of a sentence with spaCy

import spacy
nlp = spacy.load('en')

def dfs(graph, root):
    if not root in graph:
        return 1
    else:
        return max([dfs(graph, node) for node in graph[root]]) + 1

def sentence_depth(sentence):
    doc = nlp(sentence)
    sentence = [s for s in doc.sents][0]
    graph = {}
    root = None
    for word in sentence:
        head = word.head
        if head == word:
            root = head
        else:
            if not head in graph:
                graph[head] = []
            graph[head].append(word)
    return dfs(graph, root)
