from retea_neuronala import ReteaNeuronala
from procesarea_datelor import incarca_si_proceseaza_date
from evolutie import algoritm_genetic

# 1. incarcarea si prelucrarea datelor
X_train, X_test, y_train, y_test = incarca_si_proceseaza_date()

# 2. initializare retea neuronala
retea: ReteaNeuronala = ReteaNeuronala(neuroni_intrare=4, neuroni_strat_ascuns=5, neuroni_iesire=3)

# 3. rularea algoritmului evolutiv
dimensiune_populatie = 50
generatii = 100
rata_mutatie = 0.1

cel_mai_bun_cromozom, populatie_finala = algoritm_genetic(
    retea=retea,
    X_train=X_train,
    y_train=y_train,
    dimensiune_populatie=dimensiune_populatie,
    generatii=generatii,
    rata_mutatie=rata_mutatie
)

# 4. testarea si afisarea rezultatelor
predictii_train = retea.propagare_inainte(X_train, cel_mai_bun_cromozom)
predictii_test = retea.propagare_inainte(X_test, cel_mai_bun_cromozom)

print("predictii train: ", predictii_train)
print("predictii test: ", predictii_test)