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

class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(EncoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.fc_input = nn.Linear(input_size,hidden_size)
        self.fc1_hidden = nn.Linear(hidden_size,hidden_size)
        self.fc2_hidden = nn.Linear(hidden_size,hidden_size)
        self.fc_output = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x,h_prev):
        x1 = self.sigmoid((self.fc_input(x)))
        h_t = self.sigmoid(self.fc1_hidden(h_prev)+x1)
        h_t_1 = self.sigmoid(self.fc2_hidden(h_t))
        output = self.sigmoid(self.fc_output(h_t_1))
        return output
class DecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.output = nn.Linear(hidden_size,output_size)
        self.softmax = nn.sof
    def forward(self,embeddings_of_enc,hidden):
        embeddings = self.embeddings(embeddings_of_enc).view(1,-1,1)
        output,hidden = self.gru(embeddings,hidden)
        output = self.output(output[0])
        return output,hidden
    




file_name = 'file.txt'
obj_of_embeddings = Embeddings(file_name)
obj_of_pe = Positional_Encoding(file_name)

pe = obj_of_pe.positional_Encoding()
emb = obj_of_embeddings.word_embeddings()
sum_of_pe_embeddings = pe + emb
x = torch.tensor(sum_of_pe_embeddings)

obj_of_RNN = EncoderRNN(100,256,16)
h_prev = torch.zeros(256,dtype=torch.float32) # hidden size
x = x.float()

print(obj_of_RNN.forward(x,h_prev))




