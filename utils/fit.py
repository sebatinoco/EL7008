from utils.evaluate import evaluate
import torch
from torch.nn.utils import clip_grad_norm_
import os
import sys
import pickle

def fit(model, trainloader, validloader, optimizer, criterion, name, checkpoints_path, scheduler = None, epochs = 20, patience = 3, clip = 1, random_state = None):

  '''
  Función que recibe un modelo, train y valid loader, optimizador, función de pérdida y el número de epochs para luego entrenar el modelo.
  Retorna las métricas en train y valid por época.
  model: modelo a ser entrenado (torch.nn.Module)
  trainloader: datos de entrenamiento (DataLoader)
  validloader: datos de validación (DataLoader)
  optimizer: optimizador a usar (torch.nn.Optimizer)
  scheduler: scheduler a usar (torch.optim.lr_scheduler)
  criterion: función de pérdida (torch.nn.loss)
  epochs: número de épocas (int)
  patience: número máximo de épocas sin mejora en validación (int)
  name: nombre del modelo para guardarlo en disco (str)
  clip: valor máximo del gradiente en cada iteración (int)
  random_state: semilla para reproducibilidad (int)
  '''

  # Listas para visualizar métricas
  train_loss_list = [] # lista con las loss de train
  train_accuracy_list = [] # lista con los accuracy de train
  valid_loss_list = [] # lista con las loss de validation
  valid_accuracy_list = [] # lista con los accuracy de validation

  # dispositivo a usar: GPU o CPU 
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Cargamos diccionario de métricas de checkpoints
  ckpts_dict = os.path.join(checkpoints_path, 'checkpoints_dict.pkl') # ruta de diccionario
  with open(ckpts_dict, 'rb') as f:
    checkpoints_dict = pickle.load(f) # diccionario

  # Fijamos semilla 
  if random_state:
    torch.manual_seed(random_state)

  # número de batches
  n_batches = len(trainloader)

  best_accuracy_val = float('-inf')
  patience_count = 0
  # Entrenamiento
  for epoch in range(epochs): 
    model.train() # pasamos a modo "train"
    running_loss = 0.0 # loss de la época
    
    total = 0
    correct = 0
    for data in trainloader:
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device).long()

      # Fijamos a cero los gradientes
      optimizer.zero_grad()
      # Pasada hacia adelante
      outputs = model(inputs)
      # Calculamos la funcion de perdida
      loss = criterion(outputs, labels)
      # Backpropagation
      loss.backward()
      # Actualizamos los parametros con clipping
      if clip:
        clip_grad_norm_(model.parameters(), clip)
      optimizer.step()
      # Agregamos la loss de este batch
      running_loss += loss.item()

      predictions = torch.argmax(outputs, dim = 1) # obtenemos predicción usando argmax
      total += len(labels) # sumamos total de etiquetas
      correct += (predictions == labels).sum().item() # sumamos etiquetas predichas correctamente

      # Monitoreamos métricas
      sys.stdout.write(f'\rEpoch: {epoch+1:03d} \t Avg Train Loss: {running_loss/n_batches:.3f} \t Train Accuracy: {100 * correct/total:.2f} %') # print de loss promedio x epoca
    
    if scheduler:
      scheduler.step()

    # al final de la epoch, calculo métricas
    accuracy_val, loss_val, valid_total_loss = evaluate(model, validloader, criterion)

    # si tiene mejor performance, guardamos modelo en carpeta best_models
    if accuracy_val > best_accuracy_val:
     best_accuracy_val = accuracy_val
     torch.save(model, f'best_models/{name}.pt')

     patience_count = 0

    # si se supera accuracy guardado en checkpoints, exportar pesos del modelo a disco
    if accuracy_val > checkpoints_dict[name]:
      # exportamos pesos del modelo
      torch.save(model, f'{checkpoints_path}/{name}.pt')

      # sobrescribimos diccionario
      checkpoints_dict[name] = accuracy_val

      # exportamos diccionario
      with open(ckpts_dict, 'wb') as f:
        pickle.dump(checkpoints_dict, f)

    # guardamos métricas en conjuntos train y validation
    train_accuracy_list, train_loss_list = train_accuracy_list + [correct/total], train_loss_list + [running_loss/n_batches]
    valid_accuracy_list, valid_loss_list = valid_accuracy_list + [accuracy_val], valid_loss_list + [loss_val]

    # print de métricas
    if epoch % (epochs * 0.1) == 0:
      print("\t" + f"Avg Val Loss: {loss_val:.3f} \t Val Accuracy: {100 * accuracy_val:.2f} % \t Total Val Loss: {valid_total_loss:.2f}")

    model.train() # volvemos a modo entrenamiento

    # break si patience_count > patience
    patience_count += 1
    if patience_count > patience:
      break
  
  print('\n' + f'{epoch+1} épocas efectuadas')

  # guardamos modelo en carpeta folders
  torch.save(model, f'models/{name}.pt')

  return train_accuracy_list, train_loss_list, valid_accuracy_list, valid_loss_list