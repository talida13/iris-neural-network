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