'''
    fare imputazione stratificata in base alla feature ('dataset'), inoltre se la riga 
    è troppo vuota eliminare la riga e imputare le categoriche in base alla moda
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ── 1. CARICAMENTO ────────────────────────────────────────────────────────────
df = pd.read_csv(
    'C:\\Users\\pcdis\\Desktop\\Università\\BigData2\\heart_disease_uci.csv',
    sep=";"
)

# ── 2. ISPEZIONE ──────────────────────────────────────────────────────────────
print(df.isnull().mean().sort_values(ascending=False))  # % mancanti per colonna
df = df.dropna(thresh=7)                                # elimina righe troppo vuote

# ── 3. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
variabili_categoriche = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
variabili_numeriche   = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

X = df.drop(['num', 'id'], axis=1)
y = df['num']

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['dataset']
)

# ── 4. ENCODING ───────────────────────────────────────────────────────────────
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_tr[variabili_categoriche] = enc.fit_transform(X_tr[variabili_categoriche])
X_te[variabili_categoriche] = enc.transform(X_te[variabili_categoriche])

# ── 5. IMPUTAZIONE STRATIFICATA PER 'dataset' ─────────────────────────────────
def imputa_stratificato(X_tr, X_te, colonne, metodo, gruppo='dataset'):
    for col in colonne:
        if col == gruppo:
            continue
        stats = X_tr.groupby(gruppo)[col].agg(metodo)  # calcolato SOLO su X_tr
        for X in [X_tr, X_te]:
            mask = X[col].isnull()
            X.loc[mask, col] = X.loc[mask, gruppo].map(stats)
    return X_tr, X_te

X_tr, X_te = imputa_stratificato(X_tr, X_te, variabili_numeriche, 'median')
X_tr, X_te = imputa_stratificato(X_tr, X_te, variabili_categoriche,
                                  lambda x: x.mode()[0] if not x.mode().empty else np.nan)

# ── 6. SCALING ────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_tr[variabili_numeriche] = scaler.fit_transform(X_tr[variabili_numeriche])
X_te[variabili_numeriche] = scaler.transform(X_te[variabili_numeriche])

# ── 7. CORRELAZIONE ───────────────────────────────────────────────────────────
print(X_tr.corr().round(2))

# ── 8. FEATURE SELECTION + MODELLO ───────────────────────────────────────────
selector = SelectKBest(score_func=f_classif, k=10)
X_tr_sel = selector.fit_transform(X_tr, y_tr)
X_te_sel = selector.transform(X_te)

print("\nLe 10 variabili selezionate:")
print(X_tr.columns[selector.get_support()].tolist())

model = LogisticRegression(max_iter=1000)
model.fit(X_tr_sel, y_tr)
print(f"\nAccuratezza: {accuracy_score(y_te, model.predict(X_te_sel)):.2%}")