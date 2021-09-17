import math
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """Used for full context embeddings in Matching Networks."""

    def __init__(
            self,
            input_dim,
            num_layers=1,
            dropout=0,
        ):
        super().__init__()
        self.model = nn.GRU(
            input_size=input_dim, 
            hidden_size=input_dim, 
            num_layers=num_layers, 
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.input_dim = input_dim

    def forward(self, input_set):
        # input_set : batch_size x seq_len x input_dim
        output_set, _ = self.model(input_set)

        # split up the forward and backwards steps
        forward_output = output_set[:, :, :self.model.hidden_size]
        backward_output = output_set[:, :, self.model.hidden_size:]

        # Appendix A.2: g(x_i, S) = h_forward_i + h_backward_i + g'(x_i)
        output_set = forward_output + backward_output + input_set
        return output_set


class AttentionEncoder(nn.Module):
    """Used to merge support context with single example."""

    def __init__(
            self,
            input_dim,
            unrolling_steps=1,
        ):
        super().__init__()
        self.unrolling_steps = unrolling_steps
        self.cell = nn.GRUCell(input_size=input_dim, hidden_size=input_dim)

    def forward(self, support, queries):
        batch_size, num_queries, _ = queries.size()
        h_hat = torch.zeros_like(queries)

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.bmm(h, support.permute(0, 2, 1))
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.bmm(attentions, support)
            h_plus_read = h + readout

            # Run LSTM cell cf. equation (3)
            h_hat = self.cell(
                queries.view(batch_size * num_queries, -1), 
                h_plus_read.view(batch_size * num_queries, -1),
            )
            h_hat = h_hat.view(batch_size, num_queries, -1)

        h = h_hat + queries
        return h
