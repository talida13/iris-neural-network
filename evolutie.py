from ucimlrepo import fetch_ucirepo 

  
#incarcarea datelor
iris = fetch_ucirepo(id=53) 

X = iris.data.features 
y = iris.data.targets 
  
print(X)
print(y)


