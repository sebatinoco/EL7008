import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
  def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float, dropout_2: float = 0.0):

    '''
    Clase que implementa la capa de MLP referente al bloque de Encoder de ViT transformer.
    embedding_dim: Dimensión del embedding de los datos de entrada (int)
    mlp_dim: Dimensión de la capa lineal MLP para transformar los datos (int)
    dropout: Dropout a aplicar entre ambas capas lineales (float)
    dropout_2: Dropout a aplicar después de la última capa lineal (float)
    '''

    super(MultiLayerPerceptron, self).__init__()

    self.fc1 = nn.Linear(embedding_dim, mlp_dim)
    self.fc2 = nn.Linear(mlp_dim, embedding_dim)
    self.dropout = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout_2)

  def forward(self, x):

    '''
    Función que recibe un tensor de entrada, la transforma a través de capas lineales "Feed-forward".
    x: tensor a transformar (torch.tensor)
    '''

    x = self.fc1(x) # Capa 1
    x = F.gelu(x) # GELU
    x = self.dropout(x) # Dropout
    x = self.fc2(x) # Capa 2
    x = self.dropout_2(x) # Dropout

    return x    