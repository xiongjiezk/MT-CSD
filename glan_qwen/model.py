"""GLAN's executed three-branch network with frozen Qwen sentence features."""
import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        return adj @ (x @ self.weight) / adj.sum(1, keepdim=True) + self.bias


class GLANQwen(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256, hop=3, lambdaa=0.2, dropout=0.2):
        super().__init__()
        if hop < 1 or hidden_dim < 1:
            raise ValueError('hop and hidden_dim must be positive')
        self.hop, self.lambdaa = hop, lambdaa
        self.projection = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.gc1, self.gc2 = GraphConvolution(hidden_dim), GraphConvolution(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 3, 3)

    def refine(self, state, sentence):
        # Algebraically identical to the final-row masked dot products in GLAN.py.
        for _ in range(self.hop - 1):
            interaction = (sentence @ state[-1]).unsqueeze(-1) * sentence
            state = self.lambdaa * self.layer_norm1(torch.sigmoid(interaction)) + sentence
        return state[-1]

    def forward(self, utterances, spans, target_embeddings):
        x = self.dropout(self.projection(utterances))
        targets = self.projection(target_embeddings)
        outputs = []
        for i, (start, end) in enumerate(spans):
            sentence = x[start:end]
            n = len(sentence)
            if n == 0:
                raise ValueError('Empty dialogue')
            # Preserve original upper-diagonal adjacency orientation and self loops.
            adj = torch.eye(n, device=x.device, dtype=x.dtype)
            if n > 1:
                adj += torch.diag(torch.ones(n - 1, device=x.device, dtype=x.dtype), diagonal=1)
            graph = torch.tanh(self.gc2(torch.tanh(self.gc1(sentence, adj)), adj))
            local = torch.tanh(self.conv2(torch.tanh(self.conv1(sentence.T)))).T
            glob = torch.sigmoid((sentence @ sentence[-1]).unsqueeze(-1) * sentence)
            branches = torch.stack([self.refine(s, sentence) for s in (graph, local, glob)])
            weights = F.softmax(branches @ targets[i], dim=0)
            outputs.append(self.fc2((branches * weights[:, None]).flatten()))
        return torch.stack(outputs)
