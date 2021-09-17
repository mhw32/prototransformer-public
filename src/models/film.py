import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.transformer import _get_activation_fn


class FilmShiftScaleLayer(nn.Module):

    def __init__(self, d_input, d_model, d_hidden, activation="relu"):
        super(FilmShiftScaleLayer, self).__init__()

        self.gamma_fc1 = nn.Linear(d_input, d_hidden)
        self.gamma_fc2 = nn.Linear(d_hidden, d_model)
        self.beta_fc1 = nn.Linear(d_input, d_hidden)
        self.beta_fc2 = nn.Linear(d_hidden, d_model)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, task):
        gamma = self.gamma_fc2(self.activation(self.gamma_fc1(task)))
        beta = self.beta_fc2(self.activation(self.beta_fc1(task)))
        return gamma, beta
