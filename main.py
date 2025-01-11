from retea_neuronala import ReteaNeuronala
from procesarea_datelor import incarca_si_proceseaza_date
from evolutie import algoritm_genetic


# 1. incarcarea si prelucrarea datelor
X_train, X_test, y_train, y_test = incarca_si_proceseaza_date()

# 2. initializare retea neuronala
retea: ReteaNeuronala = ReteaNeuronala(neuroni_intrare=4, neuroni_strat_ascuns=5, neuroni_iesire=3)
#print(retea.dimensiune_cromozom)

# 3. rularea algoritmului evolutiv


# 4. testarea si afisarea rezultatelor





# populatie de 2 cromozomi doar pentru testarea functiilor de decodificare, mutatie, incrucisare. 
#test_cromozomi: np.array = initialize_populatie(2, retea.dimensiune_cromozom)

