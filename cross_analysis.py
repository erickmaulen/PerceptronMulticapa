from random import randrange

'''
Dividir un conjunto de datos en k pliegues
k que hace referencia al número de grupos en los que se va a dividir una muestra de datos determinada
'''
def cross_validation(dataset, num_folds):
    dataset_div = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / num_folds)
    for i in range(num_folds):
        fold = list()
        while len(fold) < fold_size:
            #Devuelve un entero aleatorio del intervalo
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_div.append(fold)
    return dataset_div


'''
Calcular el porcentaje de precision
'''
def precision(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    print(correct / float(len(actual)) * 100.0)
    return correct / float(len(actual)) * 100.0


'''Se evalua el algoritmo con una validacion cruzada
 Estimación menos sesgada
 En resumen lo que hace es lo siguiente

 1-Mezcla el conjunto de datos aleatoriamente.
 2-Dividir el conjunto de datos en grupos k
 3-Para cada grupo:
   3.1-Toma el grupo como un conjunto de datos de prueba
   3.2-Toma los grupos restantes como un conjunto de datos de entrenamiento
   3.3-Se ajusta un modelo con el conjunto de entrenamiento y se evalua en el conjunto de pruebas
   3.4-Se conserva la puntuación de evaluación y se elimina el modelo
 4- Al final la habilidad que tenga el modelo se refleja en su puntacion de evaluacion'''

def evaluate_algorithm(dataset, algorithm, num_folds, *args):
    folds = cross_validation(dataset, num_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        #La lista de pliegues se acopla en una larga lista de filas para que coincida
        #con las expectativas de los algoritmos de un conjunto de datos de entrenamiento.
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[0] for row in fold]
        exactitud = precision(actual, predicted)
        #Score de cada modelo
        scores.append(exactitud)

    return scores
