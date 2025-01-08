from ucimlrepo import fetch_ucirepo 

# incarcarea datelor
iris = fetch_ucirepo(id=53) 

X = iris.data.features 
y = iris.data.targets 
  
print(X)
print(y)

# normalizarea datelor
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_normalized = (X - X_min) / (X_max - X_min)

print("Date normalizate:")
print(X_normalized)

#impartirea datelor in set de antrenare si testare
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
