import torch
import torch.nn as nn 
import torch.nn.functional as F

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
        for layer in self.layers:
            output = F.relu(layer(output))
        return output

        

class ranker(nn.Module):

    """Docstring for ranker. """

    def __init__(self, num_features, hidden_dims = [256,128,16]):
        """TODO: to be defined. """
        nn.Module.__init__(self)
        self.mlp = MLP(num_features, hidden_dims)
    
    def forward(self, input1, input2=None):
        if input2 is None:
            return self.mlp(input1)
        else:
            return F.logsigmoid(self.mlp(input1) - self.mlp(input2))


        
