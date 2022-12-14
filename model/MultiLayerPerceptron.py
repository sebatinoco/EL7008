import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
  def __init__(self, embedding_dim, mlp_dim, dropout, dropout_2 = 0):
    super(MultiLayerPerceptron, self).__init__()

    self.fc1 = nn.Linear(embedding_dim, mlp_dim)
    self.fc2 = nn.Linear(mlp_dim, embedding_dim)
    self.dropout = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout_2)

  def forward(self, x):

    x = self.fc1(x) # Capa 1
    x = F.gelu(x) # GELU
    x = self.dropout(x) # Dropout
    x = self.fc2(x) # Capa 2
    x = self.dropout_2(x) # Dropout

    return x    