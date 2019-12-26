import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sigmoid
from torch.nn.parameter import Parameter

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Wxi = Parameter(torch.zeros(hidden_dim, input_dim))
        self.Wxf = Parameter(torch.zeros(hidden_dim, input_dim))
        self.Wxo = Parameter(torch.zeros(hidden_dim, input_dim))
        self.Wxg = Parameter(torch.zeros(hidden_dim, input_dim))

        self.Whi = Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.Whf = Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.Who = Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.Whg = Parameter(torch.zeros(hidden_dim, hidden_dim))

    def cell_step(self, x: torch.Tensor, hidden: (torch.Tensor, torch.Tensor)):

        # input shape (seq_len, batch_size, input_size(embedding size)
        # hidden is a tuple (h,c) with both c and h of shape (num_layers * num_directions, batch, hidden_size)

        h = hidden[0]
        c = hidden[1]

        i = nn.Sigmoid(self.Wxi @ x + self.Whi @ h)
        f = nn.Sigmoid(self.Wif @ x + self.Whf @ h)
        g = nn.Tanh(self.Wig @ x + self.Whg @ h)
        o = nn.Sigmoid(self.Wio @ x + self.Who @ h)
        c = torch.dot(f, c) + torch.dot(i, g)
        h = torch.dot(o, nn.Tanh(c))

        return h, c

    def forward(self, in_seq):

        h = torch.zeros_like(in_seq[0,:,:])
        c = torch.zeros_like(in_seq[0,:,:])
        outputs = []
        in_seq = in_seq.unbind(0)

        for x in in_seq:  # x is of shape (seq_len, batch_size, input_size)
            cell_out, (h, c) = self.cell_step(x, (h, c))
            outputs.append(cell_out)

        return outputs, (h, c)

def main():

    input_size = 5
    hidden_size = 10
    num_layers = 1
    output_size = 1

    lstm = LSTM(input_size, hidden_size)
    fc = nn.Linear(hidden_size, output_size)

    X = [
            [[1, 2, 3, 4, 5]],
            [[1, 2, 3, 4, 5]],
            [[1, 2, 3, 4, 5]],
            [[1, 2, 3, 4, 5]],
            [[1, 2, 3, 4, 5]],
            [[1, 2, 3, 4, 5]],
            [[1, 2, 3, 4, 5]],
    ]

    X = torch.tensor(X, dtype=torch.float32)

    print(X.shape)  # (seq_len, batch_size, input_size) = (7, 1, 5)
    out, hidden = lstm(X)  # Where X's shape is ([7,1,5])

    print('')

if __name__ == '__main__':
    main()