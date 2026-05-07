# Tutorial per l'analisi dei dati ------ BigData2
0. Nella programmazione in python, l'analisi di un dataset si baserà quasi esclusivamente nell'utilizzo della libreria **scikit-learn**

1. **Split Train/Test**
    - Obiettivo: separare i dati in due insiemi: uno per allenare il modello e uno per valutarne la performance. 
    Serve a evitare l'**overfitting** (cioè quando il modello impara solo i dati su cui è stato allenato e non generalizza).
    ```python
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y
    )
    ```
    - **`stratify`** serve a garantire che la proporzione delle classi (le etichette in y) rimanga la stessa sia nel set di train che nel set di test.   
    - Il parametro **`test_size`** indica la percentuale di dati che si vuole tenere da parte per il test (la "verifica finale").
    0.2 significa il 20%. Il restante 80% verrà usato per l'addestramento (train).
    - Il parametro **`random_state`** serve a bloccare il caso cioè, i dati vengono mescolati (shuffled) prima di essere divisi. Se non si mette un random_state:
    Ogni volta che si esegue il codice, la divisione sarà diversa.
    I tuoi risultati (accuratezza, errore) cambieranno leggermente a ogni esecuzione.
    Diventa impossibile confrontare due modelli diversi, perché non sapremmo se il miglioramento è merito del modello o solo di una divisione dei dati più "fortunata".
2. **Encoding**
    - Obiettivo: trasformare variabili non numeriche (ad esempio nomi di città o colori) in numeri, perché la maggior parte dei modelli ML lavora solo con numeri.

    - Tipi di encoding più comuni:
        - One-Hot Encoding: trasforma ogni categoria in una colonna binaria.
        - Label Encoding: assegna un numero intero a ogni categoria (utile solo per alcune situazioni, attenzione se c’è un ordine implicito).
        ```PYTHON
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        # Definizione colonne
        variabili_categoriche = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        variabili_numeriche = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        #Fit e trasformazione su training set
        X_tr_cat = enc.fit_transform(X_tr[variabili_categoriche])
        X_te_cat = enc.transform(X_te[variabili_categoriche])
        ```
3. **Imputazione**
    - Obiettivo: sostituire i valori mancanti (NaN) con valori sensati per non bloccare il modello.
    - Metodi comuni:
        - Numerici → media, mediana, valore costante
        - Categorici → moda o valore costante
4. **Feature Selection**
    - Obiettivo: scegliere le colonne più rilevanti per il modello, eliminando rumore o ridondanze. Aiuta a migliorare performance e interpretabilità.

    - Esempi in sklearn:
        - Filtraggio con correlazione (statistica)
        - SelectKBest (sceglie le migliori k feature)
        - Modelli basati su importanza (Random Forest, Gradient Boosting)
5. **Scaling(Normalizzazione)**
    - **Obiettivo**: Portare tutte le variabili numeriche sulla stessa scale (es. tra 0 e 1 o con media 0).
    -**Perché serve**: Senza scaling, il modello potrebbe dare più importanza a una colonna con numeri grandi rispetto a numeri piccoli(Reddito: 50k, Età: 30 anche se l'età fosse più importante, senza scaling il modello andrebbe a puttane).
    -**Strumenti**: `StandardScaler`(distribuzione normale) o `MinMaxScaler`(intervallo fisso).
6. **Pipeline e ColumnTransformer**

    - **Obiettivo**: Organizzare tutti i passaggi di preprocessing in un unico flusso di lavoro automatizzato e ordinato.

    - **Perché usarli: - Pulizia del codice**: Invece di avere decine di variabili temporanee (come `X_tr_cat`, `X_tr_num`, ecc.), hai un unico oggetto che gestisce tutto.

    - **Sicurezza** : Evita il "Data Leakage" (fuoriuscita di dati), assicurando che il modello non "sperimenti" mai i dati del Test Set durante la fase di apprendimento.

        **Consistenza**: Garantisce che il Test Set subisca esattamente le stesse trasformazioni (stessa media, stessa moda) calcolate sul Training Set.

    Differenza chiave:

    - `ColumnTransformer`: Si occupa della "divisione verticale" (applica trasformazioni diverse a colonne diverse).

    - `Pipeline`: Si occupa della "divisione orizzontale" (esegue i passaggi in sequenza: prima Imputazione, poi Scaling, poi Selezione).
### Documentazione
>Per la documentazione basta cercarla su internet col nome della libreria, come in questo caso sklearn.

## Sequenza imputazione valori mancanti
>1. **Caricamento** `tyu`
>2. **Ispezione** (per vedere se ci sono dati nulli)
>5. **Train test split**
>3. **Encoding numerico** delle feature categoriche (Lobial Encoder, possibile rimandarlo anche a dopo)
>4. **Imputazione** (feature per feature) [quando abbiamo colonne estremamente sparse, anche queste possono essere "scartate", se dobbiamo buttare righe o colonne meglio farlo nell'Ispezione]
Fase 3 e fase 4 possono essere scambiate
>5. **Scaling** (ogni feature viene scalata per i fatti suoi, da fare solo se possibile)
>6. **Correlazione per le feature multicollineari**
>7. **Feature selection** (score o tentativi di classificazione/clustering)
Le feature risultanti saranno profondamente diverse da quelle di partenza, le quali non vanno mai dimenticate per garantire la spiegabilità del modello usato e delle previsioni fatte (da quali feature dipendono le previsioni)

# DBSCAN CHE COS'È?

* **DBSCAN**: È un algoritmo di clustering che, a differenza di altri, non cerca di fare gruppi "rotondi", ma cerca zone ad alta densità (come stormi di uccelli). I punti che rimangono isolati li classifica come "Rumore" (assegnando loro l'etichetta `-1`).

* **Griglia di ricerca (Grid Search)**: Ti sta chiedendo di fare un ciclo `for` per provare tutte le combinazioni possibili tra i parametri forniti (n valori di Eps moltiplicati per m valori di MinPts = n*m prove in totale).
   * **Eps**: È il "raggio" visivo di un punto.
   * **MinPts**: È il numero minimo di "vicini" che un punto deve avere in quel raggio per poter formare un cluster.


# Implementazione Standard DBSCAN con Grid Search

## Codice Python Completo

```python
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import contingency_matrix
import numpy as np
import pandas as pd

#1. Griglia di ricerca
eps_values = [0.5, 1.5, 2.0]
min_pts_values = [5, 10, 20]

#2. Funzione per la valutazione
def valuta_dbscan(X, y_true, eps_val, min_pts_val, nome_set):
    #Si inizializza il DBSCAN
    db = DBSCAN(eps = eps_val, min_samples = min_pts_val)
    #Applichiamo il fit_predict 
    cluster_labels = db.fit_predict(X)
    #Calcoliamo quanti cluster utili e quanto rumore c'è 
    n_cluster = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    #Se crea 0 cluster (solo rumore), ti fermi
    if n_cluster == 0:
        print(f"[{nome_set}] Eps={eps_val}, MinPts={min_pts_val} -> Modello fallito (0 Cluster, {n_noise} punti di Rumore)")
    #Calcolo purezza
    mat = contingency_matrix(y_true, cluster_labels)
    purezza = np.sum(np.amax(mat, axis = 0))/ np.sum(mat)

    #Calcolo ACCURACY
    y_pred = np.zeros_like(cluster_labels)

    for cluster in set(cluster_labels):
        mask = (cluster_labels == cluster)
        data_in_cluster = y_true[mask]

        if len(data_in_cluster) > 0:
            # Trucco Pandas: .value_counts().idxmax() prende l'etichetta vera più frequente
            classe_maggioritaria = pd.Series(data_in_cluster).value_counts().idxmax()
            y_pred[mask] = classe_maggioritaria
            
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"[{nome_set}] Eps={eps_val}, MinPts={min_pts_val} | Cluster: {n_cluster}, Rumore: {n_noise} | Purezza: {purezza:.2%} | Accuracy: {accuracy:.2%}")

# 3. Esecuzione della griglia (i 9 tentativi)
print("=== INIZIO RICERCA PARAMETRI DBSCAN ===\n")
for e in eps_values:
    for m in min_pts_values:
        valuta_dbscan(X_tr_clean, y_tr, e, m, "TRAIN")
        valuta_dbscan(X_te_clean, y_te, e, m, "TEST ")
        print("-" * 80)
```






