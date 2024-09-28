from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize,sent_tokenize
import torch
import numpy as np
import torch.nn as nn
from gensim.utils import simple_preprocess
class Embeddings:
    def __init__(self,file_name) -> None:
        self.file_name = file_name
    def get_data(self):
        with open(self.file_name,'r') as file:
            corpus = file.readlines()
        return corpus
    def tokenization(self):
        corpus = self.get_data()
        tokens = [simple_preprocess(sentence)  for sentence in corpus ]
        return tokens
    def word_embeddings(self):
        tokens = self.tokenization()
        model = Word2Vec(sentences=tokens,vector_size=100,min_count=1,window=5)
        model.save("word2vec.model")
        Word2Vec.load("word2vec.model")
        embeddings = model.wv.vectors
        return embeddings
class Positional_Encoding(Embeddings):
    def getting_emdeddings(self):
        sentence_matrix = super().word_embeddings()
        return sentence_matrix
    def positional_Encoding(self):
        embeddings = self.getting_emdeddings()
        # For the enven encoding we use sin and for the odd we use cos
        embeddings = np.array(embeddings)
        len_seq,dim = embeddings.shape
        pos_e = np.zeros((len_seq,dim))

        k = 1
        for position in range(len_seq):
            for d in range(dim):
                if d%2==0:
                    pos_e[position, d] = np.sin(position / (10000 ** (d / dim)))
            else:
                pos_e[position,d] = np.cos(position / (10000 ** (d / dim))) 
        return pos_e
file_name = 'file.txt'
obj = Embeddings(file_name)
obj2 = Positional_Encoding(file_name)
print(obj2.positional_Encoding())

