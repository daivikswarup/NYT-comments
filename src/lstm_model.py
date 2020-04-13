import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class lstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+2, embedding_dim,\
                                      padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers =\
                            num_layers, batch_first=True)
        self.output = nn.Linear(embedding_dim, 1)
        self.pad_id = pad_id


    def forward(self, x, lengths):
        print(lengths.shape)
        padded = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                           padding_value=self.pad_id)
        print(padded.shape)
        embeddings = self.embedding(padded)
        print(embeddings)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings,lengths, \
                                                   batch_first = True,
                                                   enforce_sorted=False)
        _, (h,c) = self.lstm(packed)

        return self.output(h[-1,:,:]).squeeze(1)

