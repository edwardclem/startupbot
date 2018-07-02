'''
Character level RNN, takes N characters and predicts the next character.
'''

import numpy as np
import csv
import torch
from torch.nn.modules import LSTM
import torch.nn as nn
import string
from tqdm import tqdm, trange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


eos_marker = "<EOS>"
bos_marker = "<BOS>" #for beginning

class LinearSoftmaxLSTM(LSTM):
    '''
    LSTM with additional linear + softmax layers.
    '''
    def __init__(self, output_size, *args, **kwargs):
        super(LinearSoftmaxLSTM, self).__init__(*args, **kwargs)

        self.output_size = output_size

        #TODO: might only work if there's a single layer???
        #linear layer to get output to output_size
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def forward(self, *args, **kwargs):
        '''
        Appending the additional layers.
        '''
        #using the hidden state at the last iteration
        output, hidden_cell = super(LinearSoftmaxLSTM, self).forward(*args, **kwargs)


        hidden, cell = hidden_cell
        # print(hidden.size())

        #transposing array for Linear operation
        final_hidden = torch.transpose(hidden, 1, 0)
        # print(final_hidden.size())

        linear = self.linear(final_hidden)
        output = self.softmax(linear)

        return output, hidden_cell

    def train(self, inputs_onehot, outputs_onehot, num_data, plot_loc=None, niters=100000):
        all_losses = []

        update_every = 10

        plot_every = 1000

        current_loss = 0

        with trange(1, niters, desc='Loss: ' , leave=True) as t:
            for iter in t:
                batch_input, batch_output = random_batch(inputs_onehot, outputs_onehot, num_data)
                output, loss = train_step(self, batch_input, batch_output)
                current_loss += loss
                #TODO: change update frequency
                if iter % update_every == 0:
                    t.set_description('Loss: {}'.format(loss))

                # Add current loss avg to list of losses
                if iter % plot_every == 0:
                    all_losses.append(current_loss / plot_every)
                    current_loss = 0

        if plot_loc:
            plt.figure()
            plt.plot(all_losses)
            plt.title("training loss")
            plt.xlabel("1000 iterations")
            plt.ylabel("log-loss")
            plt.savefig(plot_loc)

    def save(self, out):
        print("Saving model to {}".format(out))
        torch.save(self.state_dict(), out)

def load_data(datafile):
    '''
    Input: filename
    output: list of strings
    Loads and extracts description data from CSV.
    '''
    data_list=[]

    with open(datafile, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_list.append(row)

    #extract all descriptions
    descriptions = [data_point['description'] for data_point in data_list]

    return descriptions

def generate_pairs(descriptions, num_chars, eos_marker, bos_marker):
    '''
    Input: list of strings, number of characters in input
    output: two lists; lists of chars of length num_chars and the following string
    Generates subsequences and outputs for RNN training. Handles EOS characters.
    '''
    #get all sets of N_CHARS + 1 letters
    inputs = []
    outputs = []
    for desc in tqdm(descriptions, desc="Generating Data Pairs"):
        #handling BOS char
        for i in range(len(desc) + 1 - num_chars):

            desc_list = [bos_marker] + list(desc)

            input_letters = desc_list[i:i + num_chars]
            inputs.append(input_letters)

            if i + num_chars + 1 == len(desc) + 1:
                #i.e. contains EOS character
                output_letter = eos_marker
            else:
                output_letter = desc_list[i + num_chars + 1]

            outputs.append(output_letter)

    return inputs, outputs


def tokenize(inputs, outputs, num_chars, vocab):
    '''
    Input: list of list of chars, list of chars
    Output: tokenized one-hot tensors for input and output
    '''
    #generate one-hot tensors for the input and output characters

    num_data = len(inputs)
    n_letters = len(vocab)

    inputs_onehot = torch.zeros(num_data, num_chars, n_letters)
    output_onehot = torch.zeros(num_data, n_letters).long()

    for i in range(len(inputs[0])):
        #computing corresponding one-hot value
        for j, letter in enumerate(inputs[i]):
            letter_index = vocab.index(letter)
            inputs_onehot[i, j, letter_index] = 1

        output_index = vocab.index(outputs[i])
        output_onehot[i, output_index] = 1

    return inputs_onehot, output_onehot

def letter_from_output(output, vocab):
    '''
    Helper function generating letters from output.
    TODO: FIX THIS
    '''
    top_n, top_i = output.topk(1)
    index_i = top_i[0][-1].item()
    return vocab[index_i], index_i

def random_batch(input, output, num_data, batch_size=10):
    '''
    Generates a random batch.
    '''

    rand_indices = np.random.choice(num_data, batch_size)

    batch_input = input[rand_indices, :, :]
    batch_output = output[rand_indices, :]
    return batch_input, batch_output

def train_step(rnn, input_batch, true_output_batch, learning_rate=0.005):
    '''
    Performs one training step of the provided network.
    '''

    rnn.zero_grad()

    output, hidden = rnn(input_batch)
    print(output)

    loss = rnn.loss(output, true_output_batch)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

def sample(rnn, vocab, num_chars, n_letters, max_length = 256, eos_marker="<EOS>"):


    last_char = ""
    chars = []

    #initialize input batch
    inputs_onehot = torch.zeros(1, num_chars, n_letters).cuda()
    inputs_onehot[:,0:(n_letters - 2),-1] = 1 #most set to bos tokens

    #set most recent to a random letter

    random_char = np.random.randint(0, n_letters)

    inputs_onehot[:,-1,random_char] = 1
    print(inputs_onehot)

    seq_len = 0
    while last_char != eos_marker and seq_len < max_length:
        output, _ = rnn(inputs_onehot)

        last_char, char_index = letter_from_output(output, vocab)

        chars.append(last_char)

        #update inputs_onehot
        inputs_onehot_new = torch.zeros(1, num_chars, n_letters).cuda()

        inputs_onehot_new[:,0:(num_chars - 1),:] = inputs_onehot[:,1:num_chars,:]
        inputs_onehot[:, -1, char_index] = 1
        seq_len += 1

    print("".join(chars))

def train_and_save(data, num_chars, vocab, rnn_loc, eos_marker = "<EOS>", bos_marker = "<BOS>", hidden_size=256, n_iters=100000):
    '''
    Trains and saves the RNN.
    '''

    # with torch.cuda.device(0):

    descriptions = load_data(data)

    inputs, outputs = generate_pairs(descriptions, num_chars, eos_marker, bos_marker)

    num_data = len(inputs)

    inputs_onehot, outputs_onehot = tokenize(inputs, outputs, num_chars, vocab)

    #initializing RNN
    #NOTE: in this case, input and output are the same size - not always going to be the case

    rnn = LinearSoftmaxLSTM(n_letters, n_letters, hidden_size, batch_first=True)
    # rnn.cuda()
    #loss function
    loss_fn = nn.NLLLoss()
    rnn.train(inputs_onehot, outputs_onehot, num_data, plot_loc="train_loss.png")
    #saves model
    rnn.save(rnn_loc)



if __name__=="__main__":

    train = True
    num_chars = 3
    rnn_loc = "../models/rnn_test"
    eos_marker = "<EOS>"
    bos_marker = "<BOS>"
    hidden_size = 256

    all_letters = list(string.ascii_letters + " .,;'-") + [eos_marker, bos_marker]
    n_letters = len(all_letters) # Plus EOS marker

    if train:
        train_and_save("../data/starthub.csv", num_chars, all_letters,rnn_loc)
    else:
        with torch.cuda.device(0):
            rnn = LinearSoftmaxLSTM(n_letters, n_letters, hidden_size, batch_first=True)
            rnn.cuda()
            rnn.load_state_dict(torch.load(rnn_loc))

            sample(rnn, all_letters, num_chars, n_letters)
