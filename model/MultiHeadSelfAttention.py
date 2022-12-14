import numpy as np
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
  def __init__(self, linear_dim, heads):
    super(MultiHeadSelfAttention, self).__init__()

    self.linear_dim = linear_dim
    self.heads = heads

    self.W_q = nn.ModuleList([nn.Linear(linear_dim, linear_dim) for _ in range(heads)]) # matriz de pesos Q
    self.W_k = nn.ModuleList([nn.Linear(linear_dim, linear_dim) for _ in range(heads)]) # matriz de pesos K
    self.W_v = nn.ModuleList([nn.Linear(linear_dim, linear_dim) for _ in range(heads)]) # matriz de pesos V

    self.W_o = nn.Linear(linear_dim * heads, linear_dim) # h x d_v

  def forward(self, x):

    output = [] # lista de resultados por head
    for head in range(self.heads):

      Q = self.W_q[head](x) # Proyección de x: Q
      K = self.W_k[head](x) # Proyección de x: K
      V = self.W_v[head](x) # Proyección de x: V

      Z = (Q @ K.transpose(1, 2)) / np.sqrt(self.linear_dim) 
      Z = torch.softmax(Z, dim = 1) # softmax
      Z = Z @ V # seq x d_v
      output.append(Z) # append de resultados
    
    output = torch.cat(output, dim = 2) # concatenamos resultados
    output = self.W_o(output) # capa lineal para volver a dimension original

    return output