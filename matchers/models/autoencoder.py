import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_layers, vocab_size, seq_len):
        super(AE, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.lstm_encoder = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    bidirectional=True)
        self.lstm_decoder = nn.LSTM(input_size=hidden_size*2,
                                    hidden_size=hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Encode input
        _, (hn, _) = self.lstm_encoder(x)

        # Concatenate left-right hidden vectors
        hn = torch.cat([hn[0], hn[1]], dim=1)

        # Reshape data to have seq_len time steps
        hn = hn.unsqueeze(0).repeat(self.seq_len, 1, 1)

        # Decode the encoded input
        output_decoder, (_, _) = self.lstm_decoder(hn)

        return self.softmax(self.dense(output_decoder))







