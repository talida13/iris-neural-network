from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def incarca_si_proceseaza_date():
    # incarcarea datelor
    iris = fetch_ucirepo(id=53) 

    X = iris.data.features 
    y = iris.data.targets 

    # normalizarea datelor
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    X_normalizat = (X - X_min) / (X_max - X_min)
    # trebuie one hot encoding pentru etichete y
    encoder = OneHotEncoder(sparse=False)
    y_encodat = encoder.fit_transform(y.values.reshape(-1, 1))

    # amestecarea datelor
    indici = np.arange(len(X_normalizat))
    np.random.seed(42)
    np.random.shuffle(indici)

    X_amestecat = X_normalizat.iloc[indici] 
    y_amestecat = y_encodat[indici]

    #impartirea datelor in set de antrenare si testare
    dimensiune_antrenare = int(0.8 * len(X_amestecat))
    X_antrenare, X_test = X_amestecat[:dimensiune_antrenare], X_amestecat[dimensiune_antrenare:]
    y_antrenare, y_test = y_amestecat[:dimensiune_antrenare], y_amestecat[dimensiune_antrenare:]

    # print("Impartirea datelor:")
    # print(X_train)
    # print(X_test)
    # print(y_train)
    # print(y_test)




