import torch
import torch.nn as nn


class RelationNetwork(nn.Module):

    def __init__(self, d_rep):
        super().__init__()
        # similarity function
        self.g = nn.Sequential(
            nn.Linear(2 * d_rep, d_rep), 
            nn.ReLU(), 
            nn.Linear(d_rep, 1),
        )
    
    def forward(self, support_features, query_features):
        """Relation Network forward pass:
        1. support_features is summed over all shots.
        2. each support_emb is then concatenated with each query_emb
        3. each concatenated vector is put through a similarity function.
        
        We return a matrix of batch_size x n_ways x (n_ways * n_query)


        Args:
        -----
        support_features: batch_size x n_ways x n_shots x d_rep
        query_features  : batch_size x n_ways x n_query x d_rep

        Source:
        -------
        https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf
        """
        batch_size, n_ways, n_query, d_rep = query_features.size()

        # support_features : batch_size x n_ways x d_rep
        support_features = support_features.sum(2)  # step 1
        
        # support_features : batch_size x (n_ways * n_query) x n_ways x d_rep
        support_features = support_features.unsqueeze(1)
        support_features = support_features.repeat(1, n_ways * n_query, 1, 1)

        # query_features : batch_size x (n_ways * n_query) x n_ways x d_rep
        query_features = query_features.view(batch_size, n_ways * n_query, d_rep)
        query_features = query_features.unsqueeze(2)
        query_features = query_features.repeat(1, 1, n_ways, 1)

        # concat_features : batch_size x (n_ways * n_query) x n_ways x (2 * d_rep)
        concat_features = torch.cat([support_features, query_features], dim=-1)

        concat_features = concat_features.view(
            batch_size * (n_ways * n_query) * n_ways,
            2 * d_rep,
        )
        # (batch_size * (n_ways * n_query) * n_ways) x 1
        scores = torch.sigmoid(self.g(concat_features))
        scores = scores.view(batch_size, n_ways * n_query, n_ways)

        return scores
