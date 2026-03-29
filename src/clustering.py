"""
=====================================================================
ÉTAPE 12 : CLUSTERING (Apprentissage Non-Supervisé)
Objectif : Regrouper les clients en segments homogènes
           sans utiliser la variable Churn (non supervisé)
Algorithme : K-Means (le plus classique et interprétable)
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ---------------------------------------------------------------
# Charger les données transformées par PCA
# On utilise X_train_pca : déjà normalisé + réduit en dimension
# Le clustering se fait sur les données d'entraînement uniquement
# ---------------------------------------------------------------
X_train_pca = pd.read_csv('data/train_test/X_train_pca.csv').values
X_test_pca  = pd.read_csv('data/train_test/X_test_pca.csv').values
y_train     = pd.read_csv('data/train_test/y_train.csv').squeeze()

print("=== Données PCA chargées ===")
print(f"X_train_pca : {X_train_pca.shape}")


# ============================================================
# ÉTAPE 12.1 : MÉTHODE DU COUDE (Elbow Method)
# Trouver le nombre optimal de clusters K
# Principe : Au-delà d'un certain K, l'inertie ne baisse plus
#            significativement → c'est le "coude"
# ============================================================

print("\n=== Recherche du K optimal (méthode du coude + silhouette) ===")
# Inertie = somme des distances entre chaque point et son centroïde
#silhouette  Mesure si chaque point est bien dans son cluster
#Étape 1 — Initialisation :
 # Placer K centroïdes aléatoirement

#Étape 2 — Assignation :
  #Chaque client rejoint le centroïde le plus proche

#Étape 3 — Mise à jour :
  #Recalculer le centroïde = moyenne de tous les points du cluster

#Étape 4 — Répéter :
  #Répéter étapes 2 et 3 jusqu'à stabilisation

inertias   = []   # Somme des distances au centroïde (à minimiser)
silhouettes = []  # Score de cohésion/séparation (à maximiser, entre -1 et 1)

K_range = range(2, 11)  # Tester de 2 à 10 clusters

for k in K_range:
    # random_state=42 pour la reproductibilité
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_train_pca)

    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_train_pca, labels, sample_size=3000, random_state=42))
    print(f"  K={k} | Inertie={km.inertia_:.0f} | Silhouette={silhouettes[-1]:.4f}")

# Visualiser les deux métriques
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(list(K_range), inertias, marker='o', color='steelblue')
axes[0].set_xlabel('Nombre de clusters K')
axes[0].set_ylabel('Inertie')
axes[0].set_title('Méthode du Coude (Elbow)')

axes[1].plot(list(K_range), silhouettes, marker='o', color='tomato')
axes[1].set_xlabel('Nombre de clusters K')
axes[1].set_ylabel('Score Silhouette')
axes[1].set_title('Score Silhouette par K')

plt.tight_layout()
plt.savefig('reports/clustering_elbow.png', dpi=120)
plt.close()
print("Graphique sauvegardé : reports/clustering_elbow.png")

# Choisir le K avec le meilleur score silhouette
best_k = list(K_range)[np.argmax(silhouettes)]
print(f"\n→ Meilleur K sélectionné : {best_k} (silhouette = {max(silhouettes):.4f})")


# ============================================================
# ÉTAPE 12.2 : ENTRAÎNER LE MODÈLE K-MEANS FINAL
# Avec le K optimal trouvé à l'étape précédente
# ============================================================

print(f"\n=== Entraînement K-Means avec K={best_k} ===")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
train_clusters = kmeans_final.fit_predict(X_train_pca)  # Assigner un cluster à chaque client
test_clusters  = kmeans_final.predict(X_test_pca)       # Prédire pour le test

# Évaluation finale
sil  = silhouette_score(X_train_pca, train_clusters, sample_size=3000, random_state=42)
dbi  = davies_bouldin_score(X_train_pca, train_clusters)
# Davies-Bouldin Index : plus petit = meilleur (clusters bien séparés)

print(f"Score Silhouette final : {sil:.4f}  (idéal proche de 1)")
print(f"Davies-Bouldin Index   : {dbi:.4f}  (idéal proche de 0)")


# ============================================================
# ÉTAPE 12.3 : VISUALISATION DES CLUSTERS EN 2D
# On utilise les 2 premières composantes PCA pour la visualisation
# ============================================================

# Charger la projection 2D
# On utilise X_train_pca directement pour la visualisation 2D
# car pca_2d a été entraîné sur les données avec feature engineering
pca_2d = joblib.load('models/pca_2d_model.pkl')

# Charger le fichier PCA complet (déjà avec toutes les features)
X_train_full = pd.read_csv('data/train_test/X_train.csv')

# Charger aussi les données avec feature engineering si elles existent
X_train_pca_full = pd.read_csv('data/train_test/X_train_pca.csv')

# Utiliser les 2 premières colonnes PCA directement pour la visualisation
# PC1 et PC2 sont déjà calculées dans X_train_pca
X_train_2d = X_train_pca[:, :2]  # Prendre juste PC1 et PC2

plt.figure(figsize=(10, 7))
colors = plt.cm.Set1(np.linspace(0, 1, best_k))

for cluster_id in range(best_k):
    mask = train_clusters == cluster_id
    plt.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
                color=colors[cluster_id], label=f'Cluster {cluster_id}',
                alpha=0.4, s=10)

# Afficher les centroïdes projetés en 2D
centroids_2d = pca_2d.transform(
    pd.DataFrame(kmeans_final.cluster_centers_,
                 columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
    .reindex(columns=X_train_full.columns, fill_value=0)
) if False else None  # Centroïdes en espace PCA complet → skip la projection ici

plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.title(f'Clustering K-Means (K={best_k}) — Projection 2D PCA')
plt.legend(markerscale=3)
plt.tight_layout()
plt.savefig('reports/clustering_2d.png', dpi=120)
plt.close()
print("Graphique clusters 2D sauvegardé : reports/clustering_2d.png")


# ============================================================
# ÉTAPE 12.4 : PROFIL / INTERPRÉTATION DES CLUSTERS
# Analyser les caractéristiques de chaque cluster
# pour leur donner un sens métier
# ============================================================

print("\n=== Profil des clusters (features clés) ===")

X_train_full = pd.read_csv('data/train_test/X_train.csv')
X_train_full['Cluster'] = train_clusters
X_train_full['Churn']   = y_train.values

# Statistiques par cluster pour les features les plus importantes
key_features = ['Recency', 'Frequency', 'MonetaryTotal', 'SatisfactionScore',
                'CustomerTenureDays', 'ReturnRatio', 'Churn']
key_features = [f for f in key_features if f in X_train_full.columns]

profile = X_train_full.groupby('Cluster')[key_features].mean().round(3)
print(profile.to_string())
profile.to_csv('reports/cluster_profiles.csv')
print("\nProfil des clusters sauvegardé : reports/cluster_profiles.csv")


#**Résultat concret :**

#         Recency  Frequency  MonetaryTotal  SatisfactionScore  Churn
#Cluster
#0          245.3       2.1         180.5            2.1        0.82  ← churners
#1           12.5      21.4        1450.2            4.7        0.04  ← fidèles
#2           88.7       7.8         520.3            3.4        0.41  ← à risque


# ============================================================
# ÉTAPE 12.5 : SAUVEGARDER LE MODÈLE ET LES LABELS
# ============================================================

joblib.dump(kmeans_final, 'models/kmeans_model.pkl')
print("\nModèle K-Means sauvegardé : models/kmeans_model.pkl")

# Sauvegarder les labels de clusters pour les réutiliser dans la classification
pd.Series(train_clusters, name='Cluster').to_csv('data/train_test/train_clusters.csv', index=False)
pd.Series(test_clusters,  name='Cluster').to_csv('data/train_test/test_clusters.csv',  index=False)
print("Labels clusters sauvegardés dans data/train_test/")