import csv
import torch
import numpy as np
from tqdm import tqdm, trange

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

    #get all unique characters

    all_chars = list(set(list("".join(descriptions))))

    return descriptions, all_chars

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

    #batch second
    inputs_onehot = torch.zeros(num_chars, num_data, n_letters)
    #NOTE: outputs not a one-hot, instead index vector
    labels = torch.zeros(num_data).long()

    for i in trange(num_data, desc="Tokenizing"):
        #computing corresponding one-hot value
        for j, letter in enumerate(inputs[i]):
            letter_index = vocab.index(letter)
            inputs_onehot[j, i, letter_index] = 1

        output_index = vocab.index(outputs[i])
        labels[i] = output_index

    return inputs_onehot, labels

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
    Batch second.
    '''

    rand_indices = np.random.choice(num_data, batch_size)

    batch_input = input[:, rand_indices, :]
    batch_output = output[rand_indices]
    return batch_input, batch_output
