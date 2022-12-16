import pandas as pd
from sklearn.model_selection import train_test_split

def gen_valsplit(metadata, class_level: str, val_size: float = .2, random_state: int = 3380):

  '''
  Función que recibe un dataframe en formato [img_name, split, high, mid, low]. Devuelve el mismo dataframe particionado en train, validation y test en la columna split
  metadata: Dataframe a particionar (pd.DataFrame)
  class_level: Clase sobre la que dividir los datos
  val_size: Porcentaje de los datos de entrenamiento que se irán a validación
  random_state: Semilla aleatoria para replicar resultadis (int)
  '''

  if class_level not in ['high', 'mid', 'low']:
    raise ValueError('Debes especificar bien el nivel de clase. Valores permitidos: high, mid, low.')

  metadata_train = metadata[metadata['split'] == 'train'].copy()
  metadata_test = metadata[metadata['split'] == 'test'].copy()

  assert metadata_train.shape[0] + metadata_test.shape[0] == metadata.shape[0]

  X_train, X_val, y_train, y_val = train_test_split(metadata_train['img_name'], metadata_train[class_level], test_size = val_size, 
                                                    stratify = metadata_train[class_level], random_state = 3380)

  X_train = pd.DataFrame(X_train)
  X_train['split'] = 'train'
  X_val = pd.DataFrame(X_val)
  X_val['split'] = 'validation'
  X = X_train.append(X_val)

  y = y_train.append(y_val)

  new_metadata = X.copy()
  new_metadata[class_level] = y
  new_metadata = new_metadata.append(metadata_test[['img_name', 'split', class_level]])
  new_metadata = new_metadata.rename(columns = {class_level: 'labels'})

  assert new_metadata.shape[0] == metadata.shape[0]

  return new_metadata