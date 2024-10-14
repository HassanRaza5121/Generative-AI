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
        self.ReLU = nn.ReLU()
    def forward(self,x,h_prev):
        x1 = self.ReLU((self.fc_input(x)))
        h_t = self.ReLU(self.fc1_hidden(h_prev)+x1)
        h_t_1 = self.ReLU(self.fc2_hidden(h_t))
        output = self.ReLU(self.fc_output(h_t_1))
        return output,h_t_1
class DecodeRNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(DecodeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.fc_input = nn.Linear(input_dim, hidden_size)
        self.fc_hidden1 = nn.Linear(hidden_size, hidden_size)
        self.fc_hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()  # Use ReLU instead of Softmax for hidden layers
        self.softmax = nn.Softmax(dim=1)  # Specify dimension for Softmax

    def forward(self, encoder_output, h_t_1):
        x1 = self.relu(self.fc_input(encoder_output))
        d_h_t = self.relu(self.fc_hidden1(h_t_1) + x1)
        d_h_t_1 = self.relu(self.fc_hidden2(d_h_t))
        output = self.softmax(self.output(d_h_t_1))
        return output, d_h_t_1

    






file_name = 'file.txt'
# Objects of All Classes

obj_of_embeddings = Embeddings(file_name)
obj_of_pe = Positional_Encoding(file_name)
obj_of_RNN = EncoderRNN(100,256,16)

pe = obj_of_pe.positional_Encoding()
emb = obj_of_embeddings.word_embeddings()
sum_of_pe_embeddings = pe + emb
x = torch.tensor(sum_of_pe_embeddings)

x = x.float()

h_prev = torch.zeros(16, 256, dtype=torch.float32)  # [batch_size, hidden_size]

# Instantiate DecoderRNN
decoder_output_dim = 100  # Example output dimension, adjust as needed
decoder = DecodeRNN(input_dim=16, hidden_size=256, output_dim=decoder_output_dim)

# Forward pass through Encoder
encoder_output, h_t_1 = obj_of_RNN(x, h_prev)
print("Encoder Output Shape:", encoder_output.shape)  # Expected: [16, 16]

# Forward pass through Decoder
decoder_output, new_h_t_1 = decoder(encoder_output, h_t_1)

print("Decoder Output Shape:", decoder_output.shape)  # Expected: [16, decoder_output_dim]












