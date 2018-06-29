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

    def forward(self, *args, **kwargs):
        '''
        Appending the additional layers.
        '''
        output_hidden, hidden = super(LinearSoftmaxLSTM, self).forward(*args, **kwargs)

        linear = self.linear(output_hidden)
        output = self.softmax(linear)

        return output, hidden


def load_data(datafile):
    '''
    Input: filename
    output: list of strings
    Loads and extracts description data from CSV.
    '''
    data_list=[]

    with open(data, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_list.append(row)

    #extract all descriptions
    descriptions = [data_point['description'] for data_point in data_list]

    return descriptions

def generate_pairs(descriptions, num_chars, eos_marker):
    '''
    Input: list of strings, number of characters in input
    output: two lists; lists of chars of length num_chars and the following string
    Generates subsequences and outputs for RNN training. Handles EOS characters.
    '''
    #get all sets of N_CHARS + 1 letters
    inputs = []
    outputs = []
    for desc in tqdm(descriptions):
        for i in range(len(desc) - NUM_CHARS):

            desc_list = list(desc)
            input_letters = desc_list[i:i + NUM_CHARS]
            inputs.append(input_letters)

            if i + NUM_CHARS + 1 == len(desc):
                #i.e. contains EOS character
                output_letter = eos_marker
            else:
                output_letter = desc_list[i + NUM_CHARS + 1]

            outputs.append(output_letter)

    return inputs, outputs


def tokenize(inputs, outputs, vocab):
    '''
    Input: list of list of chars, list of chars
    Output: tokenized one-hot tensors for input and output
    '''
    #generate one-hot tensors for the input and output characters

    num_data = len(inputs)
    n_letters = len(vocab)

    inputs_onehot = torch.zeros(num_data, NUM_CHARS, n_letters)
    output_onehot = torch.zeros(num_data, n_letters).long()

    for i in range(len(inputs[0])):
        #computing corresponding one-hot value
        for j, letter in enumerate(inputs[i]):
            letter_index = vocab.find(letter)
            inputs_onehot[i, j, letter_index] = 1

        output_index = vocab.find(outputs[i])
        output_onehot[i, output_index] = 1

    return inputs_onehot, output_onehot

def letter_from_output(output, vocab):
    '''
    Helper function generating letters from output.
    '''
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return vocab[category_i], category_i

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
    Performs one training step of the network.
    '''

    rnn.zero_grad()

    output, hidden = rnn(input_batch)

    loss = criterion(output, true_output_batch)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


if __name__=="__main__":
    NUM_CHARS = 3
    N_ITERS = 10000

    data = "../data/starthub.csv"
    eos_marker = "<EOS>"

    all_letters = string.ascii_letters + " .,;'-" + eos_marker
    n_letters = len(all_letters) # Plus EOS marker

    descriptions = load_data(data)

    inputs, outputs = generate_pairs(descriptions, NUM_CHARS, eos_marker)

    num_data = len(inputs)

    inputs_onehot, outputs_onehot = tokenize(inputs, outputs, all_letters)

    #initializing RNN
    #NOTE: in this case, input and output are the same size - not always going to be the case
    hidden_size=256
    rnn = LinearSoftmaxLSTM(n_letters, n_letters, hidden_size, batch_first=True)
    #loss
    criterion = nn.NLLLoss()

    all_losses = []

    plot_every = 1000

    current_loss = 0

    with trange(1, N_ITERS) as t:
        for iter in trange(1, N_ITERS):
            batch_input, batch_output = random_batch(inputs_onehot, outputs_onehot, num_data)
            output, loss = train_step(rnn, batch_input, batch_output)
            current_loss += loss
            #TODO: change update frequency
            t.set_description('Loss: {}'.format(loss))

            # Add current loss avg to list of losses
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0


    plt.figure()
    plt.plot(all_losses)
    plt.title("training loss")
    plt.xlabel("1000 iterations")
    plt.ylabel("log-loss")
    plt.savefig("train_loss.png")
