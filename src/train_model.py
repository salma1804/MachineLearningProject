"""
=====================================================================
ÉTAPE 13 : CLASSIFICATION — PRÉDICTION DU CHURN
Objectif : Prédire si un client va quitter l'entreprise (Churn=1)
           ou rester (Churn=0)

Modèles testés :
  1. Naive Bayes Gaussien  (probabiliste, rapide, interprétable)
  2. Random Forest         (robuste, performant, gère le déséquilibre)

Problème : Classes déséquilibrées → on gère ça avec class_weight
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ---------------------------------------------------------------
# Charger les données prétraitées + features PCA
# On va tester les modèles sur les deux espaces :
#   - X original (toutes les features scalées)
#   - X_pca     (données réduites par ACP)
# ---------------------------------------------------------------
X_train     = pd.read_csv('data/train_test/X_train.csv')
X_test      = pd.read_csv('data/train_test/X_test.csv')
X_train_pca = pd.read_csv('data/train_test/X_train_pca.csv')
X_test_pca  = pd.read_csv('data/train_test/X_test_pca.csv')
y_train     = pd.read_csv('data/train_test/y_train.csv').squeeze()
y_test      = pd.read_csv('data/train_test/y_test.csv').squeeze()

print("=== Données chargées ===")
print(f"X_train : {X_train.shape} | X_train_pca : {X_train_pca.shape}")
print(f"\nDistribution du Churn (train):")
print(y_train.value_counts(normalize=True).round(3))


# ============================================================
# ÉTAPE 13.1 : VÉRIFICATION DU DÉSÉQUILIBRE DE CLASSES
# Si Churn=1 est minoritaire, les modèles tendent à ignorer
# cette classe → class_weight='balanced' corrige ça
# ============================================================

churn_rate = y_train.mean()
print(f"\nTaux de Churn : {churn_rate:.2%}")

if churn_rate < 0.3:
    print("⚠️  Classes déséquilibrées → class_weight='balanced' sera utilisé")
    use_class_weight = 'balanced'
else:
    use_class_weight = None
    print("✓ Classes équilibrées")


# ============================================================
# FONCTION UTILITAIRE : Évaluer et afficher les résultats
# ============================================================

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    """
    Évalue un modèle de classification et retourne les métriques clés.
    Affiche le rapport de classification + matrice de confusion.
    """
    # Prédictions
    y_pred      = model.predict(X_te)
    y_pred_prob = model.predict_proba(X_te)[:, 1]  # Probabilité d'être Churn=1

    # Métriques
    report  = classification_report(y_te, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_te, y_pred_prob)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_te, y_pred))
    print(f"ROC-AUC Score : {roc_auc:.4f}")

    # Validation croisée sur le train pour détecter le surapprentissage
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc')
    print(f"CV ROC-AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fidèle', 'Churn'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'Matrice de Confusion — {name}')
    plt.tight_layout()
    plt.savefig(f'reports/confusion_{name.replace(" ","_")}.png', dpi=120)
    plt.close()

    return {'name': name, 'roc_auc': roc_auc, 'cv_mean': cv_scores.mean(),
            'precision_1': report['1']['precision'], 'recall_1': report['1']['recall'],
            'f1_1': report['1']['f1-score']}


# ============================================================
# ÉTAPE 13.2 : MODÈLE 1 — NAIVE BAYES GAUSSIEN
# Principe : Basé sur le théorème de Bayes avec l'hypothèse
#            d'indépendance conditionnelle entre les features
# Avantages : Très rapide, peu de paramètres, bonne baseline
# Limites   : Hypothèse d'indépendance souvent fausse en pratique
# NB : fonctionne mieux sur données PCA (moins de colinéarité)
# ============================================================

print("\n=== Entraînement Naive Bayes Gaussien ===")

# var_smoothing : régularisation pour éviter P(x|classe)=0
# On teste plusieurs valeurs avec GridSearchCV
gnb_params = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

gnb_base = GaussianNB()
gnb_cv   = GridSearchCV(gnb_base, gnb_params, cv=5, scoring='roc_auc', n_jobs=-1)
gnb_cv.fit(X_train_pca, y_train)  # On utilise les données PCA pour Naive Bayes

print(f"Meilleur var_smoothing : {gnb_cv.best_params_['var_smoothing']}")
gnb_best = gnb_cv.best_estimator_

# Évaluer
results_gnb = evaluate_model("Naive Bayes (PCA)", gnb_best, X_train_pca, X_test_pca, y_train, y_test)

# Sauvegarder
joblib.dump(gnb_best, 'models/naive_bayes_model.pkl')
print("Modèle Naive Bayes sauvegardé : models/naive_bayes_model.pkl")


# ============================================================
# ÉTAPE 13.3 : MODÈLE 2 — RANDOM FOREST
# Principe : Ensemble de arbres de décision entraînés sur des
#            sous-ensembles aléatoires de données et de features
# Avantages : Robuste, gère la non-linéarité, donne l'importance
#             des features, résistant au bruit
# On utilise ici les données originales (pas PCA) car RF gère
# bien la haute dimensionnalité et l'importance des features
#             est plus interprétable dans l'espace original
# ============================================================

print("\n=== Entraînement Random Forest ===")

# Hyperparamètres à tester via GridSearchCV
# Note : on réduit la grille pour limiter le temps de calcul
rf_params = {
    'n_estimators': [100, 200],          # Nombre d'arbres
    'max_depth':    [None, 10, 20],      # Profondeur max (None = complet)
    'min_samples_split': [2, 5],         # Nœuds min pour split
    'class_weight': [use_class_weight]   # Gérer le déséquilibre
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_cv   = GridSearchCV(rf_base, rf_params, cv=5, scoring='roc_auc',
                        n_jobs=-1, verbose=1)
rf_cv.fit(X_train, y_train)

print(f"Meilleurs hyperparamètres RF : {rf_cv.best_params_}")
rf_best = rf_cv.best_estimator_

# Évaluer
results_rf = evaluate_model("Random Forest", rf_best, X_train, X_test, y_train, y_test)

# Sauvegarder
joblib.dump(rf_best, 'models/random_forest_model.pkl')
print("Modèle Random Forest sauvegardé : models/random_forest_model.pkl")


# ============================================================
# ÉTAPE 13.4 : IMPORTANCE DES FEATURES (Random Forest)
# Le RF permet de savoir quelles features influencent le plus
# la prédiction du Churn → très utile pour l'interprétabilité
# ============================================================

print("\n=== Importance des features (Random Forest) ===")

importances = pd.Series(rf_best.feature_importances_, index=X_train.columns)
top20 = importances.nlargest(20)

plt.figure(figsize=(10, 7))
top20.sort_values().plot(kind='barh', color='steelblue')
plt.xlabel('Importance (Gini)')
plt.title('Top 20 features — Random Forest Churn Prediction')
plt.tight_layout()
plt.savefig('reports/rf_feature_importance.png', dpi=120)
plt.close()
print("Graphique importance features sauvegardé : reports/rf_feature_importance.png")
print("\nTop 10 features :")
print(top20.head(10).to_string())


# ============================================================
# ÉTAPE 13.5 : COURBES ROC — COMPARAISON DES MODÈLES
# La courbe ROC montre le compromis TPR/FPR pour chaque seuil
# AUC proche de 1 = excellent modèle
# AUC = 0.5 = modèle aléatoire (inutile)
# ============================================================

plt.figure(figsize=(8, 6))

# Naive Bayes
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, gnb_best.predict_proba(X_test_pca)[:, 1])
plt.plot(fpr_gnb, tpr_gnb, label=f"Naive Bayes (AUC={results_gnb['roc_auc']:.3f})", color='blue')

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_best.predict_proba(X_test)[:, 1])
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={results_rf['roc_auc']:.3f})", color='green')

# Ligne de référence (modèle aléatoire)
plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC=0.5)')

plt.xlabel('Taux Faux Positifs (FPR)')
plt.ylabel('Taux Vrais Positifs (TPR)')
plt.title('Courbes ROC — Comparaison des modèles')
plt.legend()
plt.tight_layout()
plt.savefig('reports/roc_curves.png', dpi=120)
plt.close()
print("\nCourbes ROC sauvegardées : reports/roc_curves.png")


# ============================================================
# ÉTAPE 13.6 : RÉSUMÉ COMPARATIF DES MODÈLES
# ============================================================

print("\n" + "="*60)
print("RÉSUMÉ COMPARATIF DES MODÈLES DE CLASSIFICATION")
print("="*60)

summary = pd.DataFrame([results_gnb, results_rf])
summary = summary.set_index('name')
print(summary[['roc_auc', 'cv_mean', 'precision_1', 'recall_1', 'f1_1']].round(4).to_string())
summary.to_csv('reports/classification_summary.csv')
print("\nRésumé sauvegardé : reports/classification_summary.csv")

# Identifier le meilleur modèle
best_model_name = summary['roc_auc'].idxmax()
print(f"\n→ Meilleur modèle : {best_model_name} (ROC-AUC = {summary.loc[best_model_name,'roc_auc']:.4f})")

# Sauvegarder le meilleur modèle sous un nom générique pour Flask
if best_model_name == "Random Forest":
    best_clf = rf_best
    best_X_test = X_test
else:
    best_clf = gnb_best
    best_X_test = X_test_pca

joblib.dump(best_clf, 'models/best_classifier.pkl')
print("Meilleur classificateur sauvegardé : models/best_classifier.pkl")