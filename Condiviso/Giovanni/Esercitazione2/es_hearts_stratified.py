import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

# 0. CARICAMENTO E PULIZIA INIZIALE
df = pd.read_csv("/mnt/c/Users/giova/Documents/GitHub/BigDataAnalytics/Esercitazioni/data/heart_disease_uci.csv", sep=';')

# Rimuovo le righe con 7 o più valori nulli
righe_da_eliminare = df.isnull().sum(axis=1) >= 7 #controlla il ds riga per riga
df = df[~righe_da_eliminare].copy()
print(f"Righe rimosse per eccessivi NaN: {righe_da_eliminare.sum()}")

X = df.drop(columns=['id', 'num'])
y = df['num'] # ETICHETTA ORIGINALE MULTICLASSE (0-4)

# 1. SPLIT STRATIFICATO SULLA PROVENIENZA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=X['dataset']
)

# Identifico le feature
categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()


# --- 2. SCALING ED ENCODING (PRIMA DELL'IMPUTAZIONE) ---

# SCALING
scaler = StandardScaler()
# Scaliamo i dati numerici (i NaN vengono ignorati nel calcolo e rimangono NaN)
# Li riconvertiamo subito in DataFrame mantenendo gli indici di X_train per poter fare il join dopo
X_train_num_scaled = pd.DataFrame(
    scaler.fit_transform(X_train[numerical_features]), 
    columns=numerical_features, 
    index=X_train.index
)
X_test_num_scaled = pd.DataFrame(
    scaler.transform(X_test[numerical_features]), 
    columns=numerical_features, 
    index=X_test.index
)

# ENCODING
# OneHotEncoder gestisce i NaN creando una categoria separata.
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_cat_encoded = encoder.transform(X_test[categorical_features])


# --- 3. IMPUTAZIONE INTELLIGENTE (DOPO LO SCALING) ---

# Calcoliamo le MEDIANE DI GRUPPO sui dati GIA' SCALATI del Train Set
# Usiamo .median() che è robusto agli outlier
train_grouped_stats = X_train_num_scaled.groupby(X_train['dataset']).median()

def impute_scaled_with_groupby(df_scaled, original_df, stats_df, num_cols, global_stats):
    """Riempie i NaN con la statistica (mediana) del gruppo. Se fallisce, usa quella globale."""
    df_out = df_scaled.copy()
    for col in num_cols:
        # Peschiamo la mediana di quel gruppo per questa specifica riga
        fill_values = original_df['dataset'].map(stats_df[col])
        df_out[col] = df_out[col].fillna(fill_values)
        
        # Fallback: usiamo la mediana globale se il gruppo era tutto NaN
        df_out[col] = df_out[col].fillna(global_stats[col])
    return df_out

# Mediana globale per ogni feature come rete di salvataggio
global_medians_train = X_train_num_scaled.median()

# Applichiamo la funzione usando le mediane
X_train_num_final = impute_scaled_with_groupby(
    X_train_num_scaled, X_train, train_grouped_stats, numerical_features, global_medians_train
)
X_test_num_final = impute_scaled_with_groupby(
    X_test_num_scaled, X_test, train_grouped_stats, numerical_features, global_medians_train
)

# Convertiamo in numpy array
X_train_num_final = X_train_num_final.to_numpy()
X_test_num_final = X_test_num_final.to_numpy()


# --- 4. UNIONE DEI DATI ---
X_train_final = np.hstack([X_train_num_final, X_train_cat_encoded])
X_test_final = np.hstack([X_test_num_final, X_test_cat_encoded])


# --- 5. FEATURE SELECTION E MODELLO ---
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_final, y_train)
X_test_selected = selector.transform(X_test_final)

# max_iter aumentato perché il multiclasse e le variazioni richiedono più tempo per convergere
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train_selected, y_train)

score = model.score(X_test_selected, y_test)
print(f"\nTest Accuracy (Multiclasse): {score:.2f}")
print(f"Final train shape: {X_train_selected.shape}")
print(f"Final test shape: {X_test_selected.shape}")
