# Prova scritta del 12 giugno 2024

Si consideri il data set sintetico presente nel file `data.csv` costituito da 3000 punti
con 15 feature numeriche cadauno e una label binaria di classe con valori
appartenenti a {0, 1}.

---

**1.** Caricare il dataset, estrarre randomicamente un test set pari al 5% dei dati,
mantenendo il restante 95% come training set vero e proprio. Calcolare e
stampare la matrice di correlazione tra le feature e trasformare gli eventuali
dati multicollineari con una opportuna combinazione lineare. Infine
selezionare le feature rilevanti tramite il Fisher score.

**2.** Implementare un classificatore SVM non lineare per il data set curato come
al punto precedente con i seguenti iperparametri:

- C = {1, 1/sqrt(n_samples)}
- kernel = RBF e polinomiale
- grado del kernel polinomiale = {3, 4}

Usare l'accuracy come metrica e stampare l'accuracy del miglior classificatore
e la test accuracy ottenuta in predizione.

**3.** Implementare in PyTorch una piccola rete neurale densa (fully connected) con almeno due layer lineari che esegua la classificazione binaria del data set curato, risultato del punto 1. Si utilizzi un opportuno ammontare di dropout sui layer densi ad esclusione del primo. La rete sarà addestrata con ottimizzatore RMSProp in versione "centrata" con weight decay e momento pari a 0.1. Si implementino le logiche di early stopping, destinando a validation set il 10% del training set, con una pazienza sulla validation loss di 5 epoche e un incremento minimo di miglioramento pari a 0.01; infine, si implementi un meccanismo di model checkpointing che salvi (tramite `state_dict`) solo il miglior modello valutato rispetto alla massima validation accuracy.

**4.** Confrontare i risultati dei due classificatori calcolando e stampando, per
ciascuno, la matrice di confusione, il valore di accuracy, la ROC e il valore di
AUC calcolati sul test set.