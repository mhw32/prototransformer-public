import torch.nn as nn
import torch.nn.functional as F


class TaskEmbedding(nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)



class NLPTaskEmbedding(nn.Module):

    def __init__(self, num_tasks, embedding_dim, bert_dim):
        super().__init__()
        self.fc1 = nn.Embedding(num_tasks, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, bert_dim)

    def forward(self, task):
        return self.fc2(F.relu(self.fc1(task)))

