from model.MultiHeadSelfAttention import MultiHeadSelfAttention
from model.MultiLayerPerceptron import MultiLayerPerceptron
import torch.nn as nn

class EncoderBlock(nn.Module):
  def __init__(self, embedding_dim, mlp_dim, heads, dropout, dropout_2):
    super(EncoderBlock, self).__init__()

    self.attention = MultiHeadSelfAttention(linear_dim = embedding_dim, heads = heads)
    self.mlp = MultiLayerPerceptron(embedding_dim = embedding_dim, mlp_dim = mlp_dim, dropout = dropout, dropout_2 = dropout_2)
    self.ln1 = nn.LayerNorm(embedding_dim)
    self.ln2 = nn.LayerNorm(embedding_dim)

  def forward(self, x):

    res = x # skip connection
    output = self.ln1(x) # LN
    output = self.attention(output) + res # MHSA + RES

    res = output # skip connection
    output = self.ln2(output) # LN
    output = self.mlp(output) + res # MLP + RES
        
    return output