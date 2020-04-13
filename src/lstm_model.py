import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class lstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+2, embedding_dim,\
                                      padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers =\
                            num_layers)
        self.output = nn.linear(embedding_dim, 1)
        self.pad_id = pad_id


    def forward(self, x, lengths):
        padded = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                           padding_value=self.pad_id)
        embeddings = self.embedding(padded)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings,lengths, \
                                                   enforce_sorted=False)
        _, h = self.lstm(packed)
        return self.output(h)

