import os
import json
from heapq import heappush, heappop
import numpy as np

from nltk.corpus import words as nltkwords
from nltk.corpus import wordnet

GLOVE_DIR = '/Users/stefan/glove'

def load_glove(dim):
    '''
    loads glove vectors from memory. returns map: word -> embedding
    '''
    assert dim in [50, 100, 200, 300]
    nltk_word_list = nltkwords.words()
    embeddings_index = {}
    glove_file = 'glove.6B.' + str(dim) + 'd.txt'
    f = open(os.path.join(GLOVE_DIR, glove_file))
    print 'loading glove embeddings'
    for line in f:
        values = line.split()
        word = values[0]
        if word.isalpha():
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()

    # filter using nltk words
    embeddings_filtered = dict()
    for word in nltk_word_list:
        if word in embeddings_index:
            embeddings_filtered[word] = embeddings_index[word]

    return embeddings_filtered

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

class KNN(object):
    def __init__(self, embeddings, k=5):
        self.embeddings = embeddings
        self.knn_map = dict() # maps string to list of (at most) k strings
        self.k = k
    def knn_query(self, query):
        '''
        query: string
        '''
        if query not in self.embeddings:
            return []
        if query in self.knn_map:
            return self.knn_map[query]
        knn_list = self._find_knn(query)
        self.knn_map[query] = knn_list
        return knn_list
    def _find_knn(self, query):
        query_vector = self.embeddings[query]
        h = [] # nearest neighbor heap
        for i, (candidate, vector) in enumerate(self.embeddings.items()):
            cand_dist = euclidean_distance(vector, query_vector)
            heappush(h, (cand_dist, candidate))
        knn_list = []
        for i in range(self.k):
            neighbor = heappop(h)
            knn_list.append(neighbor[1])
        return knn_list

embeddings = load_glove(50)
nn_map = {}
knn = KNN(embeddings, 50)
for i, word in enumerate(embeddings):
    if (i % 1000) == 0:
        print i
    knn.knn_query(word)
with open('neighbors.json', 'w') as fp:
    json.dump(knn.knn_map, fp)
