import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from numpy import hstack
import numpy as np

# Caricamento dati
df = pd.read_csv('C:\\Users\\pcdis\\Desktop\\Università\\BigData2\\heart_disease_uci.csv', sep=";")

#Poichè abbiamo importato dataframe -> lo dobbiamo convertire in funzione il nostro dataset separando le variabili in feature e target
#Scrivere esattamente la stessa funzione X, y = df(return_X_y = True)(che fondamentalmente si fa quando il dataset si trova dentro la libreria skit-learn) equivale a scrivere:
# Separazione target e feature
y = df['num']
X = df.drop('num', axis=1)

#1 . SPLIT TRAIN/TEST
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
    # stratify 
    # PRESERVA LA PROPORZIONE DELLE CLASSI
    # Manteniamo il dataset diviso in training (80%) e test (20%)

"""

QUESTA L'HO FATTA IO ED È PIÙ SISTEMATICA PERÒ ALCUNE FUNZIONALITÀ NON SONO DISCENDENTI DIRETTAMENTE DALLA LIBRERIA SCIKIT-LEARN 


#2 . ENCODING
        #DOBBIAMO CONVERTIRE LE VARIABILI CATEGORICHE IN NUMERI
        #UN METODO È CON L'UTILIZZO DELLA LIBRERIA DI PANDAS CHE PERò NON È AFFINE AL CORSO POICHÈ VIENE PREDILETTO L'UTILIZZO DI SKIT-LEARN 
        # X = pd.get_dummies(X)
        # DI FATTO UTILIZZEREMO LA FUNZIONE ONEHOTENCODER:
# Definizione colonne
variabili_categoriche = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
variabili_numeriche = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

#Fit e trasformazione su training set
X_tr_cat = enc.fit_transform(X_tr[variabili_categoriche])
X_te_cat = enc.transform(X_te[variabili_categoriche])

#3 . IMPUTAZIONE

#Imputazione per le var numeriche: mediana
imp_num = SimpleImputer(strategy='median')
X_tr_num = imp_num.fit_transform(X_tr[variabili_numeriche])
X_te_num = imp_num.transform(X_te[variabili_numeriche])
#Per le categoriche si può fare con la moda
imp_cat = SimpleImputer(strategy='most_frequent')
X_tr_cat = imp_cat.fit_transform(X_tr_cat)
X_te_cat = imp_cat.transform(X_te_cat)

#4 . FEATURE SELECTION
# Uniamo numeriche + categoriche dopo imputazione
X_tr_full = hstack([X_tr_num, X_tr_cat])
X_te_full = hstack([X_te_num, X_te_cat])

# Selezione delle 10 feature migliori
selector = SelectKBest(score_func=f_classif, k=10)
X_tr_selected = selector.fit_transform(X_tr_full, y_tr)
X_te_selected = selector.transform(X_te_full)

df_addestramento = pd.DataFrame(X_tr_selected)
print(df_addestramento.head())
df_test = pd.DataFrame(X_te_selected)
print(df_test.head())
#Addestramento

"""

"""VERSIONE DEFINITIVA"""
#INTEGRIAMO I PUNTI 2, 3 e 4

# Definizione colonne
variabili_categoriche = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
variabili_numeriche = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

pipe_numerica = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) #Lo scaling è caldament consigliato per f_classif
])

pipe_categorica = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), #Imputazione PRIMA
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output = False)) #Encoding DOPO
])

preprocessore = ColumnTransformer([
    ('num', pipe_numerica, variabili_numeriche),
    ('cat', pipe_categorica, variabili_categoriche)
])
# Creiamo la Pipeline finale che include anche la selezione
final_pipe = Pipeline([
    ('prep', preprocessore),
    ('select', SelectKBest(score_func=f_classif, k=10))
])

# Applichiamo tutto
X_tr_selected = final_pipe.fit_transform(X_tr, y_tr)
X_te_selected = final_pipe.transform(X_te)

print(pd.DataFrame(X_tr_selected).head())