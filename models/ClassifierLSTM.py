from torch.nn.modules import LSTM
import torch.nn as nn

class ClassifierLSTM(nn.Module):
    """
    LSTM classifier with linear + softmax layers on the end..
    """
    def __init__(self, input_size, hidden_size, label_size):
        super(ClassifierLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.input_size = input_size

        self.lstm = LSTM(input_size, hidden_size)

        self.linear = nn.Linear(hidden_size, label_size)
        # self.softmax = nn.LogSoftmax(dim=0) #summing across batch dimension


    def forward(self, sequence):
        lstm_out, _ = self.lstm(sequence)
        #using final output state
        y = self.linear(lstm_out[-1])
        # y = self.softmax(linear)
        return y
