import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    """
    Figure 2
    https://arxiv.org/pdf/1902.00751.pdf 
    """

    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.dense_down = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.dense_up = nn.Linear(hidden_size, output_size)
    
    def forward(self, hiddens, tasks):
        x = torch.cat([hiddens, tasks], dim=-1)
        resid = self.dense_down(x)
        resid = self.activation(resid)
        resid = self.dense_up(resid)
        return hiddens + resid
