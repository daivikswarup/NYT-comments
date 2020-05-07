import torch
import torch.nn as nn 
import torch.nn.functional as F
from lstm_model import lstm

class MLP(nn.Module):

    """Docstring for MLP. """

    def __init__(self, input_features, hidden_dims):
        """TODO: to be defined. """
        nn.Module.__init__(self)
        self.dims = [input_features] + hidden_dims + [1]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in
                                     zip(self.dims[:-1], self.dims[1:])])
    def forward(self, inp):
        output = inp
        for layer in self.layers[:-1]:
            output = F.relu(layer(output))
        return self.layers[-1](output)

        

class ranker(nn.Module):

    """Docstring for ranker. """

    def __init__(self, num_features, hidden_dims = []):
        """TODO: to be defined. """
        nn.Module.__init__(self)
        self.mlp = MLP(num_features, hidden_dims)
    
    def forward(self, input1, input2=None):
        if input2 is None:
            return self.mlp(input1)
        else:
            return self.mlp(input1) - self.mlp(input2)


class LSTM_ranker(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, pad_id, emb=None):
        super().__init__()
        self.model = lstm(vocab_size, embedding_size, num_layers, pad_id, emb)

    def forward(self, input1, l1, input2=None, l2=None):
        if input2 is None:
            return self.model(input1, l1)
        else:
            return self.model(input1, l1) - self.model(input2, l2)

