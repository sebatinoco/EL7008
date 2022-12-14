import torch

def evaluate(model, dataloader, criterion):

  '''
  Función que recibe un modelo y dataloader, retorna accuracy, loss promedio y loss total.
  model: clasificador (torch.nn.Module)
  dataloader: datos a ser trabajados (DataLoader)
  '''

  # dispositivo a usar: GPU o CPU 
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model.eval() # usamos modo evaluación
  with torch.no_grad(): # congelamos gradientes
      correct = 0 # inicializamos etiquetas predichas correctamente
      total = 0 # inicializamos total de etiquetas
      running_loss = 0.0
      for data in dataloader: # para cada batch del dataloader
          inputs, labels = data # obtenemos datos y etiquetas
          inputs, labels = inputs.to(device), labels.to(device).long() # pasamos a cpu o gpu
          outputs = model(inputs) # generamos predicción
          predictions = torch.argmax(outputs, dim=1) # obtenemos predicción usando argmax
          total += len(labels) # sumamos total de etiquetas
          correct += (predictions == labels).sum().item() # sumamos etiquetas predichas correctamente

          loss = criterion(outputs, labels)
          running_loss += loss.item()

  return correct/total, running_loss/len(dataloader), running_loss