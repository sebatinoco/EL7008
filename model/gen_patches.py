import torch

def gen_patches(images, patch_size: int = 4):

  batch_size, C, H, W = images.shape # dimensiones de la imagen

  assert H == W # assert imagenes cuadradas
  assert H % patch_size == 0 # assert imagenes divisibles por patch_size

  n_patches = H // patch_size # cantidad de patches en una arista

  output = [] # lista de imagenes en formato batches
  for img in images:
    batches = [] # batches de una imagen
    for row in range(n_patches):
      for col in range(n_patches):
        batches.append(img[:, patch_size * row : patch_size * (row + 1), patch_size * (col) : patch_size * (col + 1)].flatten()) # tensor (1, patch_size)

    output.append(torch.stack(batches)) 

  output = torch.stack(output)
  return output # tensor (batch_size, patches = tokens, largo)