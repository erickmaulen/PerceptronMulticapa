from csv import reader

'''
Cargar el archivo csv
'''
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        next(file)
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


'''
Convertir de string a flotante
'''
def column_to_float(dataset, column):
    for row in dataset:
        # Strip devuelve una copia de la cadena eliminando
        # los caracteres iniciales y finales (basados en el argumento de cadena pasado).
        row[column] = float(row[column].strip())

'''
Convertir de string a entero
'''
def column_to_int(dataset, column):
    class_values = [row[0] for row in dataset]
    unique = set(class_values)
    search = dict()
    for i, value in enumerate(unique):
        search[value] = i
    for row in dataset:
        row[0] = search[row[0]]
    return search


'''
Encontrar el min y max por cada columna
'''
def dataset_minmax(dataset):
    # Zip toma iterables (puede ser cero o más), los agrega en una tupla y lo devuelve.
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


'''
Normalizar los datos
'''
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(1,len(row) - 1):
            if(minmax[i][1] == 0 or minmax[i][0] == 0):
                row[i] = 0
            else:
                #print(f"{row[i]} - {minmax[i][0]} / {minmax[i][1]} - {minmax[i][0]} ")
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            #print(row[i])
