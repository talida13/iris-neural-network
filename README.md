# Antrenarea unei Rețele Neuronale de Tip Perceptron Multistrat cu un Algoritm Evolutiv

## Descriere

Acest proiect presupune antrenarea unei rețele neuronale de tip Perceptron Multistrat (MLP - Multilayer Perceptron) cu o structură predefinită (un număr specific de straturi ascunse și neuroni) utilizând un algoritm evolutiv. Algoritmul evolutiv este folosit pentru optimizarea ponderilor și pragurilor conexiunilor neuronale, astfel încât rețeaua să rezolve o problemă specifică.

Pentru acest proiect, problema aleasă este **clasificarea** folosind **setul de date Iris**, un set de date clasic utilizat frecvent în învățarea automată și statistică.

## Informații despre Setul de Date

Setul de date **Iris** conține 150 de instanțe de plante iris, împărțite în trei clase:
- **Iris Setosa**
- **Iris Versicolour**
- **Iris Virginica**

### Caracteristicile setului de date:
- **Număr de instanțe**: 150
- **Număr de caracteristici**: 4 (lungimea sepalei, lățimea sepalei, lungimea petalei, lățimea petalei) – toate sunt valori numerice continue, măsurate în centimetri.
- **Atribut de predicție**: Clasa plantei iris (Iris Setosa, Iris Versicolour sau Iris Virginica).
- **Valori lipsă**: Nu există valori lipsă.

### Observații:
- O clasă este separabilă liniar de celelalte două, în timp ce celelalte două nu sunt separabile liniar între ele.
- Setul de date este simplu și potrivit pentru evaluarea metodelor de clasificare.

## Structura Rețelei Neuronale

- **Funcția de activare**: stabilită apriori.
- **Straturi ascunse**: Se recomandă utilizarea a 1-2 straturi ascunse.
- **Număr de neuroni per strat**: Determinat în funcție de complexitatea problemei.

## Algoritmul Evolutiv

Algoritmul evolutiv va fi utilizat pentru:
1. Optimizarea ponderilor conexiunilor dintre neuroni.
2. Determinarea pragurilor pentru activarea fiecărui neuron.

## Pași în Implementare

1. **Selectarea problemei**: Problema de clasificare folosind setul de date Iris.
2. **Definirea structurii rețelei**: Numărul de straturi, neuroni și funcția de activare.
3. **Aplicarea algoritmului evolutiv**: Optimizarea parametrilor rețelei.
4. **Evaluarea performanței**: Testarea rețelei pe mulțimea de antrenare și afișarea rezultatelor.

## Rezultate

În urma procesului de antrenare, rețeaua neuronală va afișa performanța obținută pentru setul de antrenament. Acest lucru va evidenția eficiența utilizării algoritmului evolutiv pentru optimizarea rețelei.

## Resurse

- Setul de date Iris poate fi descărcat de la: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Iris).
