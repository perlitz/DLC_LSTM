import torch
import torch.nn as nn


class LangModelRNN(nn.Module):

    def __init__(self, n_tokens, embedding_dim, hidden_dim, n_layers, dropout, rnn_type):

        super(LangModelRNN, self).__init__()

        self.encoder = nn.Embedding(n_tokens, embedding_dim)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, n_tokens)

        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.init_weights()

    def forward(self, x, hidden):

        encoded = self.encoder(x)
        lstm_out, hidden = self.rnn(encoded, hidden)
        # x = x.contiguous().view(self.hidden_dim, -1)
        decoded = self.decoder(lstm_out)

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, bsz, self.hidden_dim),
                    weight.new_zeros(self.n_layers, bsz, self.hidden_dim))
        elif self.rnn_type == 'GRU':
            return weight.new_zeros(self.n_layers, bsz, self.hidden_dim)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        for hidden_parameter in self.rnn.named_parameters():
            if 'weight_hh_l' in hidden_parameter[0]:
                torch.nn.init.eye_(hidden_parameter[1])
