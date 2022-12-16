from model.EncoderBlock import EncoderBlock
from model.gen_patches import gen_patches
import torch
import torch.nn as nn

class ViT(nn.Module):
  def __init__(self, img_size: int, patch_size: int, embedding_dim: int, mlp_dim: int, n_blocks: int, heads: int, 
              n_classes: int, n_channels: int = 3, dropout: float = 0.1, dropout_2: float = 0.1):
    
    '''
    Clase que implementa el modelo ViT transformer. Se apoya de la clase EncoderBlock y la función gen_patches.
    img_size: tamaño de las imágenes de entrada (int)
    patch_size: tamaño de los patches (int)
    embedding_dim: tamaño de los embeddings a ocupar (int)
    mlp_dim: tamaño de la capa lineal del MLP a ocupar en el Encoder (int)
    n_blocks: cantidad de bloques o "capas" en el Encoder
    heads: número de "heads" por los que hacer "self-attention" en paralelo (int)
    n_classes: cantidad de clases a predecir (int)
    n_channels: número de canales de las imágenes de entrada (int)
    dropout: Dropout a aplicar entre ambas capas lineales (float)
    dropout_2: Dropout a aplicar después de la última capa lineal (float)
    '''
    
    super(ViT, self).__init__()

    self.img_size = img_size # dimension de H y W
    self.patch_size = patch_size # dimension de patches
    self.linear = nn.Linear(n_channels * self.patch_size ** 2, embedding_dim) # capa lineal de proyección de patches

    self.cls = nn.Parameter(torch.rand(1, embedding_dim)) # token cls
    self.pos = nn.Parameter(torch.rand((img_size // patch_size) ** 2 + 1, embedding_dim)) # embedding posicional

    self.blocks = nn.ModuleList(
        [EncoderBlock(embedding_dim = embedding_dim, 
                      mlp_dim = mlp_dim, 
                      heads = heads, 
                      dropout = dropout, 
                      dropout_2 = dropout_2) for _ in range(n_blocks)]) # bloques encoder

    self.header = nn.Linear(embedding_dim, n_classes) # header de clasificación

  def forward(self, x):

    '''
    Función que recibe imágenes de entrada en formato de tensor, devuelve una predicción para cada una.
    x: tensor a transformar (torch.tensor)
    '''

    batch_size, C, H, W = x.shape # dimensiones de entrada

    x = gen_patches(x, patch_size = self.patch_size) # generamos patches

    x = self.linear(x) # proyección lineal de patches

    token = self.cls.repeat(batch_size, 1, 1) # token cls
    pos = self.pos.repeat(batch_size, 1, 1) # embeddings posicionales

    x = pos + torch.cat((x, token), dim = 1) # sumamos embeddings posicionales + token cls

    # capas de atención
    for block in self.blocks:
      x = block(x)

    x = x[:, 0] # recuperamos cls token

    x = self.header(x) # header de clasificación

    return x