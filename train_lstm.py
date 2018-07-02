'''
Trains an LSTM character level language model on the provided data.
'''
from utils.data import *
from utils.train import *
from models.ClassifierLSTM import ClassifierLSTM
import torch.nn as nn
import torch.optim as optim
import string
from tqdm import tqdm, trange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__=="__main__":
    data = "data/starthub.csv"
    #set params
    num_chars = 3
    rnn_loc = "../models/rnn_test"
    eos_marker = "<EOS>"
    bos_marker = "<BOS>"
    hidden_size = 256
    niters = 10000
    plot_loc = "plots/train_loss.png"

    use_cuda = torch.cuda.is_available()

    learning_rate = 0.005

    # all_letters = list(string.ascii_letters + "& .,;'-") + ['"', eos_marker, bos_marker]
    # n_letters = len(all_letters) # Plus EOS, BOS markers


    #data processing
    descriptions, vocab = load_data(data)

    #adding EOS, BOS marker
    vocab = vocab + [eos_marker, bos_marker]

    n_letters = len(vocab)
    print("number of unique characters: {}".format(n_letters))

    inputs, outputs = generate_pairs(descriptions, num_chars, eos_marker, bos_marker)


    inputs_onehot, labels = tokenize(inputs, outputs, num_chars, vocab)

    if use_cuda:
        inputs_onehot = inputs_onehot.cuda()
        labels = labels.cuda()


    model = ClassifierLSTM(input_size=n_letters, hidden_size=hidden_size, label_size= n_letters)

    if use_cuda:
        model.cuda()

    #
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    all_losses = []

    update_every = 10

    plot_every = 1000

    current_loss = 0

    #training model
    with trange(1, niters, desc='Loss: ' , leave=True) as t:
        for iter in t:
            batch_input, batch_labels = random_batch(inputs_onehot, labels, len(inputs))
            output, loss = train_step(model, loss_fn, optimizer, batch_input, batch_labels)
            current_loss += loss

            if iter % update_every == 0:
                t.set_description('Loss: {}'.format(loss))

            # Add current loss avg to list of losses
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0


    if plot_loc:
        plt.figure()
        plt.plot(all_losses)
        plt.title("training loss w/lr = {}".format(learning_rate))
        plt.xlabel("1000 iterations")
        plt.ylabel("log-loss")
        plt.savefig(plot_loc)

    def save(self, out):
        print("Saving model to {}".format(out))
        torch.save(model.state_dict(), out)
