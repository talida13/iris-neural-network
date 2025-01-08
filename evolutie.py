from ucimlrepo import fetch_ucirepo 
import random

# fetch dataset
iris = fetch_ucirepo(id=53) 

# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

# metadata   
# print(iris.metadata) 

# variable information 
# print(iris.variables) 

print(X)
print(y)

class Selection:
    @staticmethod
    def tournament(population):
        individ1 = random.choice(population)
        individ2 = random.choice(population)

        if(individ1.fitness > individ2.fitness):
            return individ1
        else:
            return individ2

    @staticmethod
    def get_best(population):
        return max(population, key=lambda x: x.fitness)
    
class Crossover:
    @staticmethod
    def arithmetic(mother, father, rate):
        a = random.random()
        
        if random.random() >= 0.5:
            child = mother.__copy__()
        else:
            child = father.__copy__()
        
        for i in range(0, child.no_genes):
            if random.random() > rate:
                child.genes[i] = a * mother.genes[i] + (1 - a) * father.genes[i]
        
        return child

class Mutation:
    @staticmethod
    def reset(child, rate):
        if random.random() > rate:
            for i in range(0, child.no_genes):
                child.genes[i] = child.min_values[i] + random.random() * (child.max_values[i] - child.min_values[i])