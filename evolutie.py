from ucimlrepo import fetch_ucirepo 
import numpy as np

# incarcarea datelor
iris = fetch_ucirepo(id=53) 

X = iris.data.features 
y = iris.data.targets 
  
# print(X)
# print(y)

# normalizarea datelor
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_normalized = (X - X_min) / (X_max - X_min)
# print("Date normalizate:")
# print(X_normalized)

# amestecarea datelor

indices = np.arange(len(X_normalized))
np.random.seed(42)
np.random.shuffle(indices)

X_shuffled = X_normalized.iloc[indices] 
y_shuffled = y.iloc[indices] 

# print("Date amestecate:")
# print(X_shuffled)
# print(y_shuffled)


#impartirea datelor in set de antrenare si testare
train_size = int(0.8 * len(X_shuffled))
X_train, X_test = X_shuffled[:train_size], X_shuffled[train_size:]
y_train, y_test = y_shuffled[:train_size], y_shuffled[train_size:]

# print("Impartirea datelor:")
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


