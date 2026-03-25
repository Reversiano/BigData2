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




