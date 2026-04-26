# heart_disease_uci.csv
# Predisporre il dataset per l'imputazione dei valori mancanti tramite la libreria sklearn.
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Caricamento dataset
import pandas as pd
import numpy as np

# 1. Split train/test
from sklearn.model_selection import train_test_split

# 2. Encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 3. Imputazione
from sklearn.impute import SimpleImputer

# 4. Feature selection
from sklearn.feature_selection import SelectKBest, f_classif


# Caricamento
df = pd.read_csv("/mnt/c/Users/giova/Documents/GitHub/BigDataAnalytics/Esercitazioni/data/heart_disease_uci.csv", sep=';')
X = df.drop(columns=['id', 'num'])
y = df['num']

# E' necessario inizialmente effettuare uno split "stratificato" sulla feature di provenienza (4 valori di essa, di cui si hanno dati sparsi per 3 di loro)

# 1.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_features = X.select_dtypes(include=['object', 'bool']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# 2.
# Scaling Numerico
scaler = StandardScaler() #ignora i NaN durante il calcolo di mean/std
X_train_num_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_num_scaled = scaler.transform(X_test[numerical_features])
# One-Hot Encoding: gestisce i Nan creando una categoria separata
# sparse_output=False serve per poter concatenare facilmente gli array in seguito
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_cat_encoded = encoder.transform(X_test[categorical_features])

# 3.
# Imputazione Numerica (Media): riempiamo i NaN con la media della colonna, calcolata solo sui dati di training per evitare data leakage
imputer_num = SimpleImputer(strategy='mean')
X_train_num_final = imputer_num.fit_transform(X_train[numerical_features])
X_test_num_final = imputer_num.transform(X_test[numerical_features])
# Imputazione Categorica (Moda - 'most_frequent')
# È necessaria perché OneHotEncoder non gestisce i NaN di default (trasformati in una colonna binaria)
X_train_cat_final = X_train_cat_encoded
X_test_cat_final = X_test_cat_encoded

# --- UNIONE DEI DATI ---
# Concateniamo le colonne numeriche trasformate e quelle categoriche codificate
X_train_final = np.hstack([X_train_num_final, X_train_cat_final])
X_test_final = np.hstack([X_test_num_final, X_test_cat_final])

# 4.
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_final, y_train)
X_test_selected = selector.transform(X_test_final)

# Modello
model = LogisticRegression()
model.fit(X_train_selected, y_train)
score = model.score(X_test_selected, y_test)
print(f"Test Accuracy: {score:.2f}")
print(f"Final train shape: {X_train_selected.shape}")
print(f"Final test shape: {X_test_selected.shape}")
