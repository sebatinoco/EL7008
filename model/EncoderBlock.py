from model.MultiHeadSelfAttention import MultiHeadSelfAttention
from model.MultiLayerPerceptron import MultiLayerPerceptron
import torch.nn as nn

class EncoderBlock(nn.Module):
  def __init__(self, embedding_dim: int, mlp_dim: int, heads: int, dropout: float, dropout_2: float):

    '''
    Clase que implementa un bloque o "capa" del Encoder, referente al modelo ViT transformer.
    Se apoya de las clases MultiHeadSelfAttention y MultiLayerPerceptron.
    embedding_dim: dimensión del embedding de los datos de entrada (int)
    mlp_dim: Dimensión de la capa lineal MLP para transformar los datos (int)
    heads: número de "heads" por los que hacer "self-attention" en paralelo (int)
    dropout: Dropout a aplicar entre ambas capas lineales (float)
    dropout_2: Dropout a aplicar después de la última capa lineal (float)
    '''

    super(EncoderBlock, self).__init__()

    self.attention = MultiHeadSelfAttention(linear_dim = embedding_dim, heads = heads)
    self.mlp = MultiLayerPerceptron(embedding_dim = embedding_dim, mlp_dim = mlp_dim, dropout = dropout, dropout_2 = dropout_2)
    self.ln1 = nn.LayerNorm(embedding_dim)
    self.ln2 = nn.LayerNorm(embedding_dim)

  def forward(self, x):

    '''
    Función que recibe un tensor de entrada, computa "Multi-Head Self Attention" y lo transforma con capas lineales.
    x: tensor a transformar (torch.tensor)
    '''

    res = x # skip connection
    output = self.ln1(x) # LN
    output = self.attention(output) + res # MHSA + skip connection

    res = output # skip connection
    output = self.ln2(output) # LN
    output = self.mlp(output) + res # MLP + skip connection
        
    return output