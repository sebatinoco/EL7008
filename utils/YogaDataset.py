import os
from PIL import Image
from torch.utils.data import Dataset

class YogaDataset(Dataset):
  def __init__(self, root, metadata, split, transform = None):
    self.root = root
    self.split = split # especifico train o test
    self.transform = transform

    if split not in ['train', 'validation', 'test']:
      raise ValueError('Debes especificar bien el split de los datos. Valores permitidos: train, test.')

    metadata_filter = metadata[metadata['split'] == split]
    self.images = [os.path.join(root, path) for path in list(metadata_filter['img_name'])]
    self.labels = [int(label) for label in list(metadata_filter['labels'])]

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    image = Image.open(self.images[idx]) # abrimos la imagen
    label = self.labels[idx] # label de la imagen
    image = image.convert('RGB') # convertimos la imagen a RGB

    if self.transform:
      image = self.transform(image) # si se especifican transformaciones, transformar
      
    return image, label