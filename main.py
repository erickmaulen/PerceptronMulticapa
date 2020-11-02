from csv_handler import load_csv, column_to_float, column_to_int, normalize_dataset, dataset_minmax
from neuronalnet import back_propagation
from cross_analysis import evaluate_algorithm
from random import seed

seed(1)

'''
Cargar los datos
'''
filename = 'fashion-500.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    column_to_float(dataset, i)

'''
Convertir a enteros
'''
column_to_int(dataset, len(dataset[0]) - 1)

'''
Normalizar los datos
'''
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

'''
El k por defecto es 5 o 10, aunque no hay nada formal, es tanteo, eso si a medida que el k aumenta,
la diferencia de tamaño entre el conjunto de entrenamiento y los subconjuntos de testeo se hace más pequeña.
A medida que esta diferencia disminuye, el sesgo se hace más pequeño
'''
num_folds = 5
learning_rate = 0.3
epochs = 5000
num_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, num_folds, learning_rate, epochs, num_hidden)

print('Porcentajes de los modelos: %s' % scores)
print('Precision media: %.3f%%' % (sum(scores) / float(len(scores))))

