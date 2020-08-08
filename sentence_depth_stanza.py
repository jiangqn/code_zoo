# compute the depth of the dependency tree of a sentence with stanza

import stanza

nlp = stanza.Pipeline('en')

def dfs(graph, root):
    if len(graph[root]) == 0:
        return 1
    else:
        return max([dfs(graph, child) for child in graph[root]]) + 1

def sentence_depth(sentence):
    sentence = nlp(sentence)
    graph = [[] for _ in range(len(sentence.sentences[0].words) + 1)]
    for word in sentence.sentences[0].words:
        id = int(word.id)
        head = int(word.head)
        graph[head].append(id)
    return dfs(graph, 0) - 1
