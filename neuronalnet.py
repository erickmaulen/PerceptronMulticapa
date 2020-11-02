from random import random
from math import exp
from cross_analysis import precision

'''
Inicializar una red jeje
'''
def initialize_network(num_inputs, num_hidden, num_outputs):
    network = list()
    # Crear diccionario para almacenar los pesos como: [{'weights': [x x x]}, {...}, {'weights': [x x x]}]
    hidden_layer = [{'weights': [random() for i in range(num_inputs + 1)]} for i in range(num_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(num_hidden + 1)]} for i in range(num_outputs)]
    network.append(output_layer)
    return network


'''
funcion de activacion sigmoide
'''
def sigmoide(activation):
    return 1.0 / (1.0 + exp(-activation))

'''
Derivada de la funcion sigmoide
'''
def sigmoide_derivative(output):
    return output * (1.0 - output)

'''
Calcular la funcion activacion por neurona
'''
def activate(weights, inputs):

    #Añade el peso del sesgo
    activation = weights[-1]
    for i in range(len(weights) - 1):
        #suma ponderada
        activation += weights[i] * inputs[i]

    return activation


'''
Propagacion del inicio hasta el final
Todas las salidas de una capa se convierten en entradas a las neuronas de la siguiente capa'''
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoide(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


'''
Backpropagate error y almacenar en neuronas
'''
def backward_propagate_error(network, expected):
    # Se comienza de la ultima capa
    for i_layer in reversed(range(len(network))):
        layer = network[i_layer]
        errors = list()
        # Error para las capas ocultas
        if i_layer != len(network) - 1:
            for j_neurona_layer in range(len(layer)):
                error = 0.0
                # error = Sum(delta * peso enlazado con ese delta)
                for neuron in network[i_layer + 1]:
                    error = error + (neuron['weights'][j_neurona_layer] * neuron['delta'])
                errors.append(error)
                # Error calculado para la última capa
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                # diferencia de error con lo esperado
                errors.append(expected[j] - neuron['output'])
                # Almacene el error en delta para cada neurona
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoide_derivative(neuron['output'])


'''
Algoritmo backpropagation con gradiente
'''
def back_propagation(train, test, learning_rate, epochs, num_hidden):
    n_inputs = len(train[1]) - 1
    n_outputs = len(set([row[0] for row in train]))
    network = initialize_network(n_inputs, num_hidden, n_outputs)
    train_network(network, train, learning_rate, epochs, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


'''
Entrenar la red con numero fijo de epochs
learning_rate -> tasa de aprendizaje
'''
def train_network(network, train, learning_rate, epochs, num_outputs):
    for epoch in range(epochs):
        for row in train:
            # Se aplica one hot encoding, por lo tanto tiene una columna para cada valor
            # que coincida con la salida de la red, se necesita para calcular el error para la capa de salida.
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(num_outputs)]
            expected[row[0]] = 1
            # Se propaga el error
            backward_propagate_error(network, expected)
            # se actualiza los pesos
            update_weights(network, row, learning_rate)


'''
Hacer la prediccion con la red
'''
def predict(network, row):
    outputs = forward_propagate(network, row)
    #Devuelve el índice en la salida de red que tiene la mayor probabilidad
    #Se supone que los valores de clase se han convertido en enteros entre 0 y 1.
    return outputs.index(max(outputs))


'''
Actualizar los pesos de la red con el error
'''
def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            #Almacena las salidas de las capas n-1 en inputs
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            #Calcular los nuevos pesos para cada neurona de la capa n
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]

                #Actualice el sesgo de la neurona
            neuron['weights'][-1] += learning_rate * neuron['delta']
