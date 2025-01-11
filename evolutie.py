import numpy as np


# initizalizarea populatiei cu valori aleatoare
def initialize_populatie(dimensiune_populatie, dimensiune_cromozom):
    return [np.random.uniform(-1, 1, dimensiune_cromozom) for _ in range(dimensiune_populatie)]

# functia de fitness
def fitness(cromozom, retea, X_train, y_train):
    return "De facut"

# selectia parintilor
def selectie_parinti(populatie, fitnessuri):
    return "De facut"


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
def algoritm_genetic(retea, X_train, y_train, dimensiune_populatie=50, generatii=100, rata_mutatie=0.1):
    return "de facut"








