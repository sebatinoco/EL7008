import matplotlib.pyplot as plt

def graph_metrics(train_loss_list: list, train_accuracy_list: list, valid_loss_list: list, valid_accuracy_list: list, interval = 5):

  '''
  Función que recibe listas con métricas de loss y accuracy para conjuntos de train y validación, y los grafica.
  train_loss_list: lista con las métricas de loss del conjunto train (list)
  train_accuracy_list: lista con las métricas de accuracy del conjunto train (list)
  valid_loss_list: lista con las métricas de loss del conjunto valid (list)
  valid_accuracy_list: lista con las métricas de accuracy del conjunto valid (list)
  interval: ticks para marcar en eje x (int)
  '''

  epochs = len(train_loss_list)

  fig, axes = plt.subplots(1, 2, figsize = (16, 4)) # configuramos tamaño del gráfico
  xticks = range(1, epochs + 1, interval - 1) # xticks

  # grafico de loss
  axes[0].plot(range(1, epochs + 1), [i for i in train_loss_list], label = 'Train', linewidth = 2) # plot de loss en train
  axes[0].plot(range(1, epochs + 1), [i for i in valid_loss_list], label = 'Valid', linewidth = 2) # plot de loss en train
  axes[0].set(title = 'Loss Promedio vs Epochs', xlabel = 'Epochs', ylabel = 'Loss promedio', xticks = xticks) # labels grafico 0
  axes[0].legend() # mostramos leyendas

  # grafico de accuracy
  axes[1].plot(range(1, epochs + 1), [i for i in train_accuracy_list], label = 'Train', linewidth = 2) # plot de accuracy en train
  axes[1].plot(range(1, epochs + 1), [i for i in valid_accuracy_list], label = 'Valid', linewidth = 2) # plot de accuracy en train
  axes[1].set(title = 'Accuracy vs Epochs', xlabel = 'Epochs', ylabel = 'Accuracy', xticks = xticks) # labels grafico 1
  axes[1].legend() # mostramos leyendas

  plt.show()