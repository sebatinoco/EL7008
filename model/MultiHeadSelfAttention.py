import numpy as np
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
  def __init__(self, linear_dim: int, heads: int):

    '''
    Clase que implementa el "Multi-Head Self Attention" de la arquitectura transformers. 
    linear_dim: dimensión lineal de los datos de entrada (int)
    heads: número de "heads" por los que hacer "self-attention" en paralelo (int)
    '''

    super(MultiHeadSelfAttention, self).__init__()

    self.linear_dim = linear_dim
    self.heads = heads

    self.W_q = nn.ModuleList([nn.Linear(linear_dim, linear_dim) for _ in range(heads)]) # matriz de pesos Q
    self.W_k = nn.ModuleList([nn.Linear(linear_dim, linear_dim) for _ in range(heads)]) # matriz de pesos K
    self.W_v = nn.ModuleList([nn.Linear(linear_dim, linear_dim) for _ in range(heads)]) # matriz de pesos V

    self.W_o = nn.Linear(linear_dim * heads, linear_dim) # h x d_v

  def forward(self, x):

    '''
    Función que recibe un tensor de entrada, computa "Multi-Head Self Attention" sobre este. 
    Devuelve un tensor con las mismas dimensiones.
    x: tensor a transformar (torch.tensor)
    '''

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