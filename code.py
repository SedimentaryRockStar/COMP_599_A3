import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import pandas as pd
from matplotlib import pyplot as plt

device = torch.device("mps" if torch.has_mps else "cpu")

##### PROVIDED CODE #####

def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]

def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}

def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in tokenize(batch['premise'] + batch['hypothesis']):
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


##### END PROVIDED CODE #####

class CharSeqDataloader():
    def __init__(self, filepath, seq_len, examples_per_epoch):
        self.unique_chars = []

        with open(filepath, 'r') as f:
            self.dataset= ''.join([line for line in f.readlines()])


        for ch in self.dataset:
            if ch not in self.unique_chars:
                self.unique_chars.append(ch)
        self.vocab_size = len(self.unique_chars)
        self.mappings = self.generate_char_mappings(self.unique_chars)
        # Now Process The whole dataset to indices and pass it to the device
        self.dataset= torch.tensor(self.convert_seq_to_indices(self.dataset)).to(device)

        self.seq_len = seq_len
        self.examples_per_epoch = examples_per_epoch





    def generate_char_mappings(self, uq):
        char_to_idx= {}
        idx_to_char= {}
        for i, char in zip(range(len(uq)), uq):
            char_to_idx[char]= i
            idx_to_char[i]= char
        return {"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}

    def convert_seq_to_indices(self, seq):
        return [ self.mappings["char_to_idx"][char] for char in seq]

    def convert_indices_to_seq(self, seq):
        return [ self.mappings["idx_to_char"][idx] for idx in seq]

    def get_example(self):
        for _ in range(self.examples_per_epoch):
            seq_idx= random.randint(0, len(self.dataset)- self.seq_len- 1)
            yield self.dataset[seq_idx: seq_idx+ self.seq_len], self.dataset[seq_idx+ 1: seq_idx+ self.seq_len+ 1]



class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars
        self.embedding_size = embedding_size

        self.embedding_layer= nn.Embedding(self.n_chars, self.embedding_size, device= device)

        self.wax= nn.Linear(self.embedding_size, self.hidden_size, device= device, bias= True)
        self.waa= nn.Linear(self.hidden_size, self.hidden_size, device= device, bias=False)
        self.wya= nn.Linear(self.hidden_size, self.n_chars, device= device, bias= True)

    def rnn_cell(self, i, h):
        h_new= torch.tanh(self.wax(i)+ self.waa(h))
        return self.wya(h_new), h_new


    def forward(self, input_seq, hidden = None):
        embs= self.embedding_layer(input_seq)
        tensor_list= []
        for idx, emb in enumerate(embs):
            hidden= hidden if hidden is not None else torch.zeros(self.hidden_size, requires_grad=False)
            o, hidden= self.rnn_cell(emb, hidden)
            tensor_list.append(o)

        return torch.stack(tensor_list).to(device), hidden


    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(),lr= lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        seq= [starting_char]
        feed= self.embedding_layer(torch.tensor(starting_char))
        hidden= torch.zeros(self.hidden_size, requires_grad=False)
        for _ in range(seq_len):
            feed, hidden= self.rnn_cell(feed, hidden)
            m = Categorical(F.softmax( feed/temp, dim= 0))
            feed= m.sample()
            seq.append(feed.item())
            feed= self.embedding_layer(feed)
        return seq

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size, device=device)

        self. forget_gate= nn.Linear(self.embedding_size+ self.hidden_size, self.hidden_size,bias= True)
        self. input_gate= nn.Linear(self.embedding_size+ self.hidden_size, self.hidden_size,bias= True)
        self. output_gate= nn.Linear(self.embedding_size+ self.hidden_size, self.hidden_size,bias= True)
        self. cell_state_layer= nn.Linear(self.embedding_size+ self.hidden_size, self.hidden_size,bias= True)
        self. fc_output=  nn.Linear(self.hidden_size, self.n_chars, bias= True)

    def forward(self, input_seq, hidden = None, cell = None):
        embs = self.embedding_layer(input_seq)
        tensor_list = []
        for idx, emb in enumerate(embs):
            hidden = hidden if hidden is not None else torch.zeros(self.hidden_size, requires_grad=False)
            cell = cell if cell is not None else torch.zeros(self.hidden_size, requires_grad=False)
            o, hidden, cell = self.lstm_cell(emb, hidden, cell)
            tensor_list.append(o)

        return torch.stack(tensor_list).to(device), hidden, cell

    def lstm_cell(self, i, h, c):
        cat= torch.concat((i, h), dim= 0)
        ft= torch.sigmoid(self.forget_gate(cat))
        it= torch.sigmoid(self.input_gate(cat))
        tct= torch.tanh(self.cell_state_layer(cat))
        c_new= ft * c+ it* tct
        ot= torch.sigmoid(self.output_gate(cat))
        h_new= ot* torch.tanh(c_new)

        return self.fc_output(h_new), h_new, c_new

    def get_loss_function(self):
        return nn.CrossEntropyLoss()


    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        seq = [starting_char]
        feed = self.embedding_layer(torch.tensor(starting_char))
        hidden = torch.zeros(self.hidden_size, requires_grad=False)
        cell = torch.zeros(self.hidden_size, requires_grad=False)
        for _ in range(seq_len):
            feed, hidden, cell = self.lstm_cell(feed, hidden, cell)
            m = Categorical(F.softmax(feed / temp, dim=0))
            feed = m.sample()
            seq.append(feed.item())
            feed = self.embedding_layer(feed)
        return seq

def train(model, dataset, lr, out_seq_len, epoch_size, num_epochs):

    loss= model.get_loss_function()
    optim= model.get_optimizer(lr)
    model.to(device)
    n = 0
    running_loss = 0
    loss_list= []
    for epoch in range(num_epochs):
        for in_seq, out_seq in dataset.get_example():
            sequence, _, _= model.forward(in_seq)
            optim.zero_grad()
            output= loss(sequence, F.one_hot(out_seq, num_classes= len(dataset.unique_chars)).type(torch.DoubleTensor).to(device)).requires_grad_().to(device)

            output.backward()
            optim.step()
            running_loss += output.item()
            n += 1
        loss_list.append(running_loss/n)
        # print info every X examples
        print(f"Epoch {epoch}. Running loss so far: {(running_loss/n):.8f}")

        print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from your model randomly

        with torch.no_grad():
            starting_char= random.randint(0, len(dataset.unique_chars)- 1)
            seq= model.sample_sequence(starting_char, out_seq_len, temp= 1)
            #print(seq)
            print("".join([dataset.mappings['idx_to_char'][idx] for idx in seq]))
        print("\n------------/SAMPLE FROM MODEL/------------")

        n = 0
        running_loss = 0

    plt.plot(loss_list)
    plt.show()
    return model


def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.001
    num_epochs = 1000
    epoch_size = 50 # one epoch is this # of examples
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    dataset= CharSeqDataloader(data_path, seq_len, epoch_size)
    model= CharRNN(dataset.vocab_size, embedding_size, hidden_size)

    
    train(model, dataset, lr=lr,
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs,
                epoch_size= epoch_size)

    idx_I = dataset.mappings["char_to_idx"]['I']
    for i in np.arange(0.1, 1.1, 0.1):
        seq = (model.sample_sequence(idx_I, 100, temp= i))
        print(f"\ntemperature now is {i: .2f}")
        print("".join([dataset.mappings['idx_to_char'][idx] for idx in seq]))

def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 1000
    epoch_size = 50
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    dataset= CharSeqDataloader(data_path, seq_len, epoch_size)
    model = CharLSTM(dataset.vocab_size, embedding_size, hidden_size)
    train(model, dataset, lr=lr,
                 out_seq_len=out_seq_len,
                 num_epochs=num_epochs,
                 epoch_size=epoch_size), dataset.mappings["char_to_idx"]['I']

    idx_I = dataset.mappings["char_to_idx"]['I']
    for i in np.arange(0.1, 1.1, 0.1):
        seq = (model.sample_sequence(idx_I, 100, temp=i))
        print(f"\ntemperature now is {i: .2f}")
        print("".join([dataset.mappings['idx_to_char'][idx] for idx in seq]))


def fix_padding(batch_premises, batch_hypotheses):
    bat_premises= []
    bat_hypotheses= []
    bat_premises_rev = []
    bat_hypotheses_rev = []
    for ex in batch_premises:
        bat_premises.append(torch.tensor(ex))
        bat_premises_rev.append(torch.tensor(ex[::-1]))
    for ex in batch_hypotheses:
        bat_hypotheses.append(torch.tensor(ex))
        bat_hypotheses_rev.append(torch.tensor(ex[::-1]))
    p= torch.nn.utils.rnn.pad_sequence(bat_premises, batch_first= True).type(torch.IntTensor).to(device)
    h= torch.nn.utils.rnn.pad_sequence(bat_hypotheses, batch_first= True).type(torch.IntTensor).to(device)
    p_r= torch.nn.utils.rnn.pad_sequence(bat_premises_rev, batch_first= True).type(torch.IntTensor).to(device)
    h_r= torch.nn.utils.rnn.pad_sequence(bat_hypotheses_rev, batch_first= True).type(torch.IntTensor).to(device)
    return p, h, p_r, h_r

def create_embedding_matrix(word_index, emb_dict, emb_dim):
    emb_matrix= torch.zeros((len(emb_dict), emb_dim))
    for word in word_index:
        emb_matrix[word_index[word]]= torch.from_numpy(emb_dict[word]) if word in emb_dict else torch.zeros(emb_dim)
    return emb_matrix


def evaluate(model, dataloader, index_map):
    counter = 0
    length= 0
    for i, sample in enumerate(dataloader):
        premises= tokens_to_ix(tokenize(sample['premise']), index_map)
        hypotheses= tokens_to_ix(tokenize(sample['hypothesis']), index_map)
        #print(premises, hypotheses)
        if len(premises[0])== 0 or len(hypotheses[0])== 0:
            continue
        result = model.forward(premises, hypotheses)
        predictions = result.argmax(dim=-1)
        counter+= torch.sum((predictions== sample['label'])).item()
        length+= 1
    return counter/length


class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embed_dim, num_layers, num_classes, embeddings):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_dim= embed_dim

        #self.embedding_layer= nn.Embedding(vocab_size, embed_dim, padding_idx= 0)
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze= False, padding_idx= 0)
        self.int_layer= nn.Linear(2* hidden_dim, hidden_dim, bias= True)
        self.out_layer= nn.Linear(hidden_dim, num_classes, bias= True)
        self.lstm= nn.LSTM(input_size= embed_dim, hidden_size= hidden_dim, batch_first= True, num_layers= num_layers)

    def forward(self, a, b):
        p, h, _, _= fix_padding(a, b)
        _, (_, cp)= self.lstm(self.embedding_layer(p))
        _, (_, ch) = self.lstm(self.embedding_layer(h))
        return self.out_layer(torch.relu(self.int_layer(torch.cat([cp, ch], dim= -1)))).squeeze()

class TrueLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embed_dim, num_layers, num_classes, embeddings):
        super(TrueLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_dim= embed_dim

        #self.embedding_layer= nn.Embedding(vocab_size, embed_dim, padding_idx= 0)
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze= False, padding_idx= 0)
        self.int_layer= nn.Linear(4* hidden_dim, hidden_dim, bias= True)
        self.out_layer= nn.Linear(hidden_dim, num_classes, bias= True)
        self.lstm= nn.LSTM(input_size= embed_dim, hidden_size= hidden_dim, batch_first= True, num_layers= num_layers,
                           bidirectional= True)

    def forward(self, a, b):
        p, h, _, _= fix_padding(a, b)
        _, (_, cp)= self.lstm(self.embedding_layer(p))
        _, (_, ch) = self.lstm(self.embedding_layer(h))
        return self.out_layer(torch.relu(self.int_layer(torch.cat([cp[-1,:,:], ch[-1,:,:], cp[-2,:,:], ch[-2,:,:]], dim= -1)))).squeeze()

class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embed_dim, num_layers, num_classes, embeddings):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        #self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        self.int_layer = nn.Linear(4 * hidden_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, num_classes, bias=True)
        self.lstm_forward = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers)
        self.lstm_backward = nn.LSTM(input_size= embed_dim, hidden_size=hidden_dim, batch_first=True,
                                    num_layers=num_layers)


    def forward(self, a, b):
        p, h, p_r, h_r= fix_padding(a, b)

        _, (_, cp)= self.lstm_forward(self.embedding_layer(p))
        _, (_, cp_r) = self.lstm_backward(self.embedding_layer(p_r))
        _, (_, ch) = self.lstm_forward(self.embedding_layer(h))
        _, (_, ch_r) = self.lstm_backward(self.embedding_layer(h_r))
        return self.out_layer(torch.relu(self.int_layer(torch.cat([cp, cp_r, ch, ch_r], dim= -1)))).squeeze()


def run_snli(model):
    print(device)
    dataset = load_dataset("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col= 0)

    glove_dict= glove.T.to_dict('list') # Convert the dataframe to a dictionary style


    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered =  dataset['test'].filter(lambda ex: ex['label'] != -1)

    #partial= list(range(0, 10))
    #train_filtered= Subset(train_filtered, partial)
    dataloader_train= DataLoader(train_filtered, batch_size= 32, shuffle= True)
    dataloader_valid= DataLoader(valid_filtered)
    dataloader_test= DataLoader(test_filtered)
    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    for key in glove_dict:
        glove_dict[key]= np.array(glove_dict[key])
    glove_embeddings = create_embedding_matrix(index_map, glove_dict, 100)

    lstm= model(vocab_size= glove_embeddings.size()[0], hidden_dim= 100, embed_dim= 100, num_layers= 1, num_classes= 3, embeddings= glove_embeddings)

    loss= nn.CrossEntropyLoss()
    optim= torch.optim.Adam(lstm.parameters(), lr= 0.002)

    lstm.to(device)
    n = 0
    running_loss = 0
    train_loss_list = []
    valid_loss_list= []
    test_loss_list= []
    for epoch in range(10):
        print(f"Now epoch {epoch}")
        for i, sample in enumerate(dataloader_train):
            premises = tokens_to_ix(tokenize(sample['premise']), index_map)
            hypotheses = tokens_to_ix(tokenize(sample['hypothesis']), index_map)
            #print(premises, hypotheses)
            result = lstm.forward(premises, hypotheses)

            optim.zero_grad()
            output = loss(result,
                          F.one_hot(sample['label'], num_classes= 3).squeeze().type(torch.FloatTensor).to(device)).requires_grad_().to(device)
            output.backward()
            optim.step()
            running_loss += output.item()
            n += 1
            #print(f'Current {i} Running Loss: {running_loss / n}')

        train_loss_list.append(running_loss / n)
        n = 0
        running_loss = 0
        with torch.no_grad():
            valid= evaluate(lstm, dataloader_valid, index_map)
            test= evaluate(lstm, dataloader_test, index_map)
            valid_loss_list.append(valid)
            test_loss_list.append(test)
            print(valid, test)
    plt.plot(valid_loss_list, label= "Validating Loss")
    plt.plot(test_loss_list, label= "Testing Loss")

    plt.legend()
    plt.show()

    return lstm




def run_snli_lstm():
    model_class =  UniLSTM
    run_snli(model_class)

def run_snli_bilstm():
    model_class = ShallowBiLSTM
    run_snli(model_class)

def run_snli_truelstm():
    model_class = TrueLSTM
    run_snli(model_class)

if __name__ == '__main__':
    run_snli_truelstm()

    #run_char_rnn()

    #run_char_lstm()
    #run_snli_lstm()
    # run_snli_bilstm()
