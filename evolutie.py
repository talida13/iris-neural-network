from ucimlrepo import fetch_ucirepo 
import numpy as np

# reteaua neuronala
neuroni_intrare = 4  # numarul de caracteristici din setul Iris (sepal length, sepal width, petal length, petal width)
neuroni_strat_ascuns = 5  # numarul de neuroni din stratul ascuns
neuroni_iesire = 3  # numarul de clase din setul Iris (setosa, versicolor, virginica)

nr_ponderi_ascuns = neuroni_intrare * neuroni_strat_ascuns
nr_praguri_ascuns = neuroni_strat_ascuns
nr_ponderi_iesire = neuroni_strat_ascuns * neuroni_iesire
nr_praguri_iesire = neuroni_iesire

dimensiune_cromozom = nr_ponderi_ascuns + nr_praguri_ascuns + nr_ponderi_iesire + nr_praguri_iesire

#initizalizarea populatiei cu valori aleatoare
def initialize_populatie(dimensiune_populatie, dimensiune_cromozom):
    return [np.random.uniform(-1, 1, dimensiune_cromozom) for _ in range(dimensiune_populatie)]

# decodificare cromozom


# functia de propagare inainte


# functia de fitness


# selectia parintilor


# incrucisarea



# mutatia



# algoritmul genetic


# antrenarea retelei neuronale

# incarcarea datelor
iris = fetch_ucirepo(id=53) 

X = iris.data.features 
y = iris.data.targets 

# normalizarea datelor
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_normalizat = (X - X_min) / (X_max - X_min)

# amestecarea datelor
indici = np.arange(len(X_normalizat))
np.random.seed(42)
np.random.shuffle(indici)

X_amestecat = X_normalizat.iloc[indici] 
y_amestecat = y.iloc[indici] 

#impartirea datelor in set de antrenare si testare
dimensiune_antrenare = int(0.8 * len(X_amestecat))
X_antrenare, X_test = X_amestecat[:dimensiune_antrenare], X_amestecat[dimensiune_antrenare:]
y_antrenare, y_test = y_amestecat[:dimensiune_antrenare], y_amestecat[dimensiune_antrenare:]
# print("Impartirea datelor:")
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Functia sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


