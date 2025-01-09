from ucimlrepo import fetch_ucirepo 
import numpy as np
from retea_neuronala import ReteaNeuronala

# initializare retea neuronala
retea: ReteaNeuronala = ReteaNeuronala(neuroni_intrare=4, neuroni_strat_ascuns=5, neuroni_iesire=3)
#print(retea.dimensiune_cromozom)

# initizalizarea populatiei cu valori aleatoare
def initialize_populatie(dimensiune_populatie, dimensiune_cromozom):
    return [np.random.uniform(-1, 1, dimensiune_cromozom) for _ in range(dimensiune_populatie)]

# populatie de 2 cromozomi doar pentru testarea functiilor de decodificare, mutatie, incrucisare. 
#test_cromozomi: np.array = initialize_populatie(2, retea.dimensiune_cromozom)

# decodificare cromozom
def decodificare_cromozom(cromozom: np.array, retea: ReteaNeuronala):
    ponderi_ascuns = cromozom[:retea.nr_ponderi_ascuns].reshape((retea.neuroni_intrare, retea.neuroni_strat_ascuns))
    
    praguri_ascuns = cromozom[retea.nr_ponderi_ascuns:retea.nr_ponderi_ascuns + retea.nr_praguri_ascuns]

    inceput_iesire = retea.nr_ponderi_ascuns + retea.nr_praguri_ascuns

    ponderi_iesire = cromozom[inceput_iesire:inceput_iesire + retea.nr_ponderi_iesire].reshape((retea.neuroni_strat_ascuns, retea.neuroni_iesire))
    
    praguri_iesire = cromozom[inceput_iesire + retea.nr_praguri_iesire:]

    return {
        "ponderi_ascuns": ponderi_ascuns,
        "praguri_ascuns": praguri_ascuns,
        "ponderi_iesire": ponderi_iesire,
        "praguri_iesire": praguri_iesire
    }

# apel pentru a testa functionalitatea.
#decodificare = decodificare_cromozom(test_cromozomi[0], retea)
#print(decodificare["ponderi_ascuns"])
#print(decodificare)


# functia de propagare inainte


# functia de fitness


# selectia parintilor


# incrucisarea
def incrucisare(parinte1, parinte2):
    punct = np.random.randint(1, len(parinte1) - 1)
    copil1 = np.concatenate((parinte1[:punct], parinte2[punct:]))
    copil2 = np.concatenate((parinte2[:punct], parinte1[punct:]))
    return copil1, copil2

# apel de functie pentru testarea functionalitatii.
#print(incrucisare(test_cromozomi[0], test_cromozomi[1]))

# mutatia
def mutatie(cromozom, rata_mutatie=0.1):
    for i in range(len(cromozom)):
        if np.random.rand() < rata_mutatie:
            cromozom[i] += np.random.uniform(-0.5, 0.5)
    return cromozom

# apel de functie pentru testarea functionalitatii.
#print(mutatie(test_cromozomi[0]))

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


