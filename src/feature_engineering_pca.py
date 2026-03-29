"""
=====================================================================
ÉTAPE 10 : FEATURE ENGINEERING
Création de nouvelles features à partir des features existantes
pour capturer plus d'information comportementale
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# Charger les données déjà préprocessées (issues du script précédent)
# On recharge X_train et X_test depuis les fichiers sauvegardés
# ---------------------------------------------------------------
X_train = pd.read_csv('data/train_test/X_train.csv')
X_test  = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv').squeeze()  # squeeze() → Series
y_test  = pd.read_csv('data/train_test/y_test.csv').squeeze()

print("=== Données chargées ===")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")


# ============================================================
# ÉTAPE 10.1 : FEATURE ENGINEERING
# Créer de nouvelles variables combinant des features existantes
# pour enrichir l'information donnée au modèle
# ============================================================

def add_engineered_features(df):
    """
    Ajoute des features construites à partir des features existantes.
    Appelée sur train et test séparément (pas de fuite de données).
    """

    # --- Ratio dépense / récence ---
    # Un client qui dépense beaucoup mais récemment = valeur élevée
    # +1 pour éviter la division par zéro si Recency = 0
    #recency = nb de jour des le derniere achat (9adeh tada w howa machra chy 9ad ma recency faible kad ma client actif )
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    # --- Panier moyen par commande ---
    # Montant total divisé par le nombre de commandes
    
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)

    # --- Ratio ancienneté / récence ---
    # Un client ancien qui n'achète plus récemment = risque de churn élevé
    #client depuis 1000 jours mais n'a pas acheté depuis 900 jours
#           → TenureRatio proche de 1 → DANGER de churn !
    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)

    # --- Taux d'annulation par commande ---
    # Nombre d'annulations rapporté au nombre total de transactions
    if 'CancelledTransactions' in df.columns and 'Frequency' in df.columns:
        df['CancelRate'] = df['CancelledTransactions'] / (df['Frequency'] + 1)

    # --- Score d'engagement global ---
    # Fréquence élevée + satisfaction élevée + peu de support tickets = client engagé
    if all(c in df.columns for c in ['Frequency', 'SatisfactionScore', 'SupportTicketsCount']):
        df['EngagementScore'] = (df['Frequency'] * df['SatisfactionScore']) / (df['SupportTicketsCount'] + 1)

    return df

# Appliquer le feature engineering sur train et test
X_train = add_engineered_features(X_train)
X_test  = add_engineered_features(X_test)

print(f"\n=== Après Feature Engineering ===")
print(f"X_train: {X_train.shape}")
print(f"Nouvelles features créées: MonetaryPerDay, AvgBasketValue, TenureRatio, CancelRate, EngagementScore")


# ============================================================
# ÉTAPE 10.2 : RE-SCALING des nouvelles features
# Les nouvelles features créées ne sont pas encore normalisées
# On les scaler séparément avec un nouveau StandardScaler
# ============================================================

new_features = ['MonetaryPerDay', 'AvgBasketValue', 'TenureRatio', 'CancelRate', 'EngagementScore']
# Ne garder que celles effectivement créées
new_features = [f for f in new_features if f in X_train.columns]

scaler_new = StandardScaler()
#fit transform :on calcule le moyen et l'ecart type par contre dans trasforn elles sont deja calculees
X_train[new_features] = scaler_new.fit_transform(X_train[new_features])   # fit sur train uniquement(val-moy/ecart_type)
X_test[new_features]  = scaler_new.transform(X_test[new_features])        # transform sur test

print(f"Nouvelles features scalées : {new_features}")


# ============================================================
# ÉTAPE 11 : ACP (Analyse en Composantes Principales / PCA)
# Objectif : réduire la dimensionnalité tout en conservant
#            le maximum d'information (variance expliquée)
# Utile pour : visualisation, clustering, réduction du bruit
# ============================================================

print("\n=== ACP / PCA ===")

# --- Étape 11.1 : Trouver le nombre optimal de composantes ---
# On lance d'abord une PCA avec TOUTES les composantes
# pour voir combien d'entre elles expliquent 95% de la variance

pca_full = PCA(random_state=42)
pca_full.fit(X_train)  # fit UNIQUEMENT sur X_train (éviter la fuite de données) yamel fl pc1,pc2.....

# Variance cumulée expliquée y7seb fl varience 
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Nombre de composantes pour expliquer 95% de la variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Nombre de composantes pour 95% de variance : {n_components_95}")
print(f"Variance expliquée par les 10 premières composantes : {cumulative_variance[9]:.3f}")

# --- Visualisation de la variance expliquée ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, 21), pca_full.explained_variance_ratio_[:20], color='steelblue')
plt.xlabel('Composante')
plt.ylabel('Variance expliquée')
plt.title('Variance par composante (Top 20)')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='.', color='tomato')
plt.axhline(y=0.95, color='green', linestyle='--', label='Seuil 95%')
plt.axvline(x=n_components_95, color='orange', linestyle='--', label=f'{n_components_95} composantes')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance cumulée')
plt.title('Variance cumulée expliquée')
plt.legend()

plt.tight_layout()
plt.savefig('reports/pca_variance.png', dpi=120)
plt.close()
print("Graphique sauvegardé : reports/pca_variance.png")


# --- Étape 11.2 : PCA finale avec le nombre optimal de composantes ---
pca = PCA(n_components=n_components_95, random_state=42)
X_train_pca = pca.fit_transform(X_train)   # fit+transform sur train
X_test_pca  = pca.transform(X_test)        # transform seulement sur test

print(f"\nX_train après PCA : {X_train_pca.shape}")
print(f"X_test  après PCA : {X_test_pca.shape}")


# --- Étape 11.3 : PCA 2D pour visualisation des clusters futurs ---
# On crée une version 2D séparée, uniquement pour la visualisation
pca_2d = PCA(n_components=2, random_state=42)
X_train_2d = pca_2d.fit_transform(X_train)

plt.figure(figsize=(8, 6))
# Colorier par la variable Churn pour voir la séparation
scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                      c=y_train, cmap='coolwarm', alpha=0.4, s=10)
plt.colorbar(scatter, label='Churn (0=Fidèle, 1=Parti)')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.title('Projection 2D PCA — colorée par Churn')
plt.tight_layout()
plt.savefig('reports/pca_2d_churn.png', dpi=120)
plt.close()
print("Graphique PCA 2D sauvegardé : reports/pca_2d_churn.png")


# --- Étape 11.4 : Sauvegarder les données transformées par PCA ---
# Nommer les colonnes PC1, PC2, ...
pca_cols = [f'PC{i+1}' for i in range(n_components_95)]

X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols)
X_test_pca_df  = pd.DataFrame(X_test_pca,  columns=pca_cols)

X_train_pca_df.to_csv('data/train_test/X_train_pca.csv', index=False)
X_test_pca_df.to_csv('data/train_test/X_test_pca.csv',   index=False)

print(f"\nDonnées PCA sauvegardées :")
print(f"  data/train_test/X_train_pca.csv")
print(f"  data/train_test/X_test_pca.csv")


# --- Sauvegarder les objets PCA pour les réutiliser en production ---
import joblib
import os
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

joblib.dump(pca,       'models/pca_model.pkl')
joblib.dump(pca_2d,    'models/pca_2d_model.pkl')
joblib.dump(scaler_new,'models/scaler_new_features.pkl')

print("\nModèles PCA sauvegardés dans models/")
print("  models/pca_model.pkl")
print("  models/pca_2d_model.pkl")