"""
=====================================================================
ÉTAPE 14 : RÉGRESSION — PRÉDICTION DU MONTANT TOTAL DÉPENSÉ
Objectif : Prédire la variable continue MonetaryTotal
           (combien un client va dépenser au total)

Modèles testés :
  1. Régression Ridge   (régularisation L2, gère la multicolinéarité)
  2. Random Forest Reg. (non-linéaire, robuste aux outliers)

Métriques : R², RMSE, MAE
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ---------------------------------------------------------------
# Charger les données — la cible est maintenant MonetaryTotal
# On doit retirer MonetaryTotal des features et l'utiliser comme y
# ---------------------------------------------------------------
X_train_full = pd.read_csv('data/train_test/X_train.csv')
X_test_full  = pd.read_csv('data/train_test/X_test.csv')

# Vérifier que MonetaryTotal est présente
if 'MonetaryTotal' not in X_train_full.columns:
    print("⚠️  MonetaryTotal absente — utilisation de Recency comme cible de démonstration")
    target_col = 'Recency'
else:
    target_col = 'MonetaryTotal'

print(f"=== Régression : prédiction de {target_col} ===")

# Séparer X et y pour la régression
# On exclut aussi Churn car c'est une variable cible de classification
cols_to_drop = [target_col]

y_train_reg = X_train_full[target_col].copy()
y_test_reg  = X_test_full[target_col].copy()

X_train_reg = X_train_full.drop(columns=cols_to_drop)
X_test_reg  = X_test_full.drop(columns=cols_to_drop)

print(f"X_train_reg : {X_train_reg.shape}")
print(f"y_train_reg — Moyenne : {y_train_reg.mean():.2f}, Std : {y_train_reg.std():.2f}")


# ============================================================
# FONCTION UTILITAIRE : Évaluer un modèle de régression
# ============================================================

def evaluate_regression(name, model, X_tr, X_te, y_tr, y_te):
    """
    Évalue un modèle de régression.
    Affiche R², RMSE, MAE + plot prédictions vs réalité.
    """
    y_pred = model.predict(X_te)

    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)

    # Validation croisée R² sur train
    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2')

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"R²   (test)  : {r2:.4f}   (1.0 = parfait, 0 = modèle nul)")
    print(f"RMSE (test)  : {rmse:.4f}  (erreur en £ — à minimiser)")
    print(f"MAE  (test)  : {mae:.4f}  (erreur absolue moyenne)")
    print(f"CV R² (train): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # Graphique : Valeurs réelles vs Prédictions
    plt.figure(figsize=(7, 5))
    plt.scatter(y_te, y_pred, alpha=0.3, s=10, color='steelblue')
    # Ligne parfaite y=x
    mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='Prédiction parfaite')
    plt.xlabel(f'Valeurs réelles ({target_col})')
    plt.ylabel('Valeurs prédites')
    plt.title(f'{name} — Réel vs Prédit\nR²={r2:.3f}, RMSE={rmse:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports/regression_{name.replace(" ","_")}.png', dpi=120)
    plt.close()

    return {'name': name, 'r2': r2, 'rmse': rmse, 'mae': mae, 'cv_r2': cv_r2.mean()}


# ============================================================
# ÉTAPE 14.1 : MODÈLE 1 — RÉGRESSION RIDGE
# Principe : Régression linéaire avec pénalité L2 sur les coefficients
# La pénalité évite le surapprentissage et gère la multicolinéarité
# Paramètre clé : alpha (plus alpha est grand, plus la régularisation est forte)
# ============================================================

print("\n=== Entraînement Ridge Regression ===")

ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}
ridge_base   = Ridge()
ridge_cv     = GridSearchCV(ridge_base, ridge_params, cv=5, scoring='r2', n_jobs=-1)
ridge_cv.fit(X_train_reg, y_train_reg)

print(f"Meilleur alpha Ridge : {ridge_cv.best_params_['alpha']}")
ridge_best = ridge_cv.best_estimator_

results_ridge = evaluate_regression("Ridge Regression", ridge_best,
                                    X_train_reg, X_test_reg, y_train_reg, y_test_reg)

# Coefficients Ridge — interpréter les features les plus influentes
coef = pd.Series(np.abs(ridge_best.coef_), index=X_train_reg.columns)
top_coef = coef.nlargest(15)

plt.figure(figsize=(9, 6))
top_coef.sort_values().plot(kind='barh', color='mediumseagreen')
plt.title('Top 15 coefficients Ridge (valeur absolue)')
plt.xlabel('|Coefficient|')
plt.tight_layout()
plt.savefig('reports/ridge_coefficients.png', dpi=120)
plt.close()
print("Graphique coefficients Ridge sauvegardé : reports/ridge_coefficients.png")


# ============================================================
# ÉTAPE 14.2 : MODÈLE 2 — RANDOM FOREST REGRESSOR
# Avantage : Capture les relations non-linéaires
#            et les interactions entre features
# ============================================================

print("\n=== Entraînement Random Forest Regressor ===")

rfr_params = {
    'n_estimators': [100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5]
}

rfr_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rfr_cv   = GridSearchCV(rfr_base, rfr_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
rfr_cv.fit(X_train_reg, y_train_reg)

print(f"Meilleurs hyperparamètres RF Regressor : {rfr_cv.best_params_}")
rfr_best = rfr_cv.best_estimator_

results_rfr = evaluate_regression("Random Forest Regressor", rfr_best,
                                  X_train_reg, X_test_reg, y_train_reg, y_test_reg)


# ============================================================
# ÉTAPE 14.3 : RÉSUMÉ ET SAUVEGARDE
# ============================================================

print("\n" + "="*60)
print("RÉSUMÉ COMPARATIF — RÉGRESSION")
print("="*60)

summary_reg = pd.DataFrame([results_ridge, results_rfr]).set_index('name')
print(summary_reg[['r2', 'rmse', 'mae', 'cv_r2']].round(4).to_string())
summary_reg.to_csv('reports/regression_summary.csv')
print("\nRésumé sauvegardé : reports/regression_summary.csv")

# Meilleur modèle de régression = celui avec R² le plus élevé
best_reg_name = summary_reg['r2'].idxmax()
print(f"\n→ Meilleur modèle de régression : {best_reg_name} (R²={summary_reg.loc[best_reg_name,'r2']:.4f})")

# Sauvegarder les deux modèles
joblib.dump(ridge_best, 'models/ridge_regression_model.pkl')
joblib.dump(rfr_best,   'models/rf_regressor_model.pkl')
print("Modèles de régression sauvegardés dans models/")