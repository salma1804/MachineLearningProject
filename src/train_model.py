import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model  import LogisticRegression, Ridge
from sklearn.ensemble      import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster       import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics       import silhouette_score

import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from utils import (
    evaluate_classifier, evaluate_regressor,
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, detect_overfitting,
    compare_models, find_optimal_k, run_kmeans,
    describe_clusters, plot_pca_2d
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, '..')
TT_DIR    = os.path.join(ROOT_DIR, 'data', 'train_test')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
REP_DIR   = os.path.join(ROOT_DIR, 'reports')

for d in [MODEL_DIR, REP_DIR]:
    os.makedirs(d, exist_ok=True)


def load_data():
    print("\n" + "="*65)
    print("  CHARGEMENT DES DONNÉES")
    print("="*65)

    X_train     = pd.read_csv(os.path.join(TT_DIR, 'X_train.csv'))
    X_test      = pd.read_csv(os.path.join(TT_DIR, 'X_test.csv'))
    y_train     = pd.read_csv(os.path.join(TT_DIR, 'y_train.csv')).values.ravel()
    y_test      = pd.read_csv(os.path.join(TT_DIR, 'y_test.csv')).values.ravel()
    X_train_pca = pd.read_csv(os.path.join(TT_DIR, 'X_train_pca.csv'))
    X_test_pca  = pd.read_csv(os.path.join(TT_DIR, 'X_test_pca.csv'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

    print(f"  X_train       : {X_train.shape}")
    print(f"  X_test        : {X_test.shape}")
    print(f"  X_train_pca   : {X_train_pca.shape}")
    print(f"  y_train dist  : {dict(pd.Series(y_train).value_counts().sort_index())}")
    print(f"  y_test  dist  : {dict(pd.Series(y_test).value_counts().sort_index())}")

    return X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, feature_names



def run_clustering(X_train_pca, X_train_original):
    print("\n" + "="*65)
    print("  CLUSTERING — K-MEANS")
    print("="*65)

    find_optimal_k(X_train_pca.values, k_range=range(2, 10), save_dir=REP_DIR)

  
    n_clusters = 4
    labels, kmeans = run_kmeans(X_train_pca.values,
                                n_clusters=n_clusters, save_dir=REP_DIR)

    sil = silhouette_score(X_train_pca.values, labels, sample_size=2000,
                           random_state=42)
    print(f"\n  Score silhouette (k={n_clusters}) : {sil:.4f}")
    print("  (1=parfait, 0=clusters chevauchés, <0=mauvais)")

    describe_clusters(X_train_original, labels, save_dir=REP_DIR)


    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans.pkl'))
    print(f"\n  kmeans.pkl sauvegardé")

    return labels, kmeans, sil



def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n" + "="*65)
    print("  CLASSIFICATION — LOGISTIC REGRESSION + GridSearchCV")
    print("="*65)

    params = {
        'C':            [0.001, 0.01, 0.1, 1],
        'penalty':      ['l2'],
        'solver':       ['lbfgs', 'liblinear'],
        'class_weight': ['balanced', None]
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid=params,
        cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print(f"\n  Meilleurs paramètres : {grid.best_params_}")
    print(f"  Meilleur F1 (CV)     : {grid.best_score_:.4f}")

    metrics, y_pred, y_proba = evaluate_classifier(best, X_test, y_test,
                                                    'LogisticRegression')
    detect_overfitting(best, X_train, y_train, X_test, y_test)

    plot_confusion_matrix(
        y_test, y_pred, 'LogisticRegression',
        save_path=os.path.join(REP_DIR, 'cm_logreg.png')
    )
    plot_roc_curve(
        y_test, y_proba, 'LogisticRegression',
        save_path=os.path.join(REP_DIR, 'roc_logreg.png')
    )

    joblib.dump(best, os.path.join(MODEL_DIR, 'logreg.pkl'))
    return best, metrics


def train_random_forest_classifier(X_train, y_train, X_test, y_test, feature_names):
    print("\n" + "="*65)
    print("  CLASSIFICATION — RANDOM FOREST + GridSearchCV")
    print("="*65)

    params = {
        'n_estimators':      [100, 200],
        'max_depth':         [5, 10, 15],
        'class_weight':      ['balanced', None],
        'min_samples_split': [5, 10],
        'min_samples_leaf':  [2, 4],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=params,
        cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print(f"\n  Meilleurs paramètres : {grid.best_params_}")
    print(f"  Meilleur F1 (CV)     : {grid.best_score_:.4f}")

    metrics, y_pred, y_proba = evaluate_classifier(best, X_test, y_test,
                                                    'RandomForest')
    detect_overfitting(best, X_train, y_train, X_test, y_test)

    plot_confusion_matrix(
        y_test, y_pred, 'RandomForest',
        save_path=os.path.join(REP_DIR, 'cm_rf.png')
    )
    plot_roc_curve(
        y_test, y_proba, 'RandomForest',
        save_path=os.path.join(REP_DIR, 'roc_rf.png')
    )
    plot_feature_importance(
        best, feature_names, top_n=15,
        model_name='RandomForest',
        save_path=os.path.join(REP_DIR, 'feature_importance_rf.png')
    )

    joblib.dump(best, os.path.join(MODEL_DIR, 'random_forest.pkl'))
    return best, metrics



def train_regression(X_train_raw_path, X_test_raw_path,
                     target_col='MonetaryTotal'):
    """
    Entraîne un modèle de régression pour prédire MonetaryTotal
    à partir des features comportementales et RFM.

    On utilise ici les données scaled (sans SMOTE, car SMOTE est
    réservé à la classification binaire).
    """
    print("\n" + "="*65)
    print(f"  RÉGRESSION — Prédiction de {target_col}")
    print("="*65)

    proc_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        'data', 'processed', 'retail_customers_CLEANED.csv'
    )
    if not os.path.exists(proc_path):
        print("  Fichier processed introuvable — régression ignorée.")
        return None, None

    df_reg = pd.read_csv(proc_path)
    if target_col not in df_reg.columns:
        print(f"  Colonne {target_col} absente — régression ignorée.")
        return None, None

    drop_cols = ['Churn', target_col]
    X_reg = df_reg.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
    y_reg = df_reg[target_col]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler

    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    sc = StandardScaler()
    X_r_train = sc.fit_transform(X_r_train)
    X_r_test  = sc.transform(X_r_test)

    params = {
        'n_estimators': [100, 200],
        'max_depth':    [None, 10],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=params,
        cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
    )
    grid.fit(X_r_train, y_r_train)

    best = grid.best_estimator_
    print(f"\n  Meilleurs paramètres : {grid.best_params_}")

    metrics, y_pred = evaluate_regressor(best, X_r_test, y_r_test,
                                          f'RF-Regressor ({target_col})')

    joblib.dump(best, os.path.join(MODEL_DIR, 'regressor.pkl'))
    print(f"  regressor.pkl sauvegardé")
    return best, metrics


def select_best_classifier(results, X_test, y_test):
    """
    Compare LogReg vs RandomForest sur F1 et AUC-ROC.
    Sauvegarde le meilleur sous best_model.pkl.
    """
    print("\n" + "="*65)
    print("  COMPARAISON DES MODÈLES DE CLASSIFICATION")
    print("="*65)

    compare_models(
        results, metric='f1',
        save_path=os.path.join(REP_DIR, 'model_comparison_f1.png')
    )
    compare_models(
        results, metric='auc_roc',
        save_path=os.path.join(REP_DIR, 'model_comparison_auc.png')
    )

    best_name = max(results, key=lambda k: results[k]['f1'])
    print(f"\n  Meilleur modèle (F1) : {best_name}  "
          f"F1={results[best_name]['f1']:.4f}  "
          f"AUC={results[best_name]['auc_roc']:.4f}")

    models_map = {
        'LogisticRegression': os.path.join(MODEL_DIR, 'logreg.pkl'),
        'RandomForest':       os.path.join(MODEL_DIR, 'random_forest.pkl'),
    }
    best_model = joblib.load(models_map[best_name])
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
    print(f"  best_model.pkl sauvegardé → {best_name}")
    return best_model, best_name


def main():
    print("\n" + "="*65)
    print("   PIPELINE DE MODÉLISATION — PROJET ML RETAIL")
    print("="*65)
    (X_train, X_test, y_train, y_test,
     X_train_pca, X_test_pca, feature_names) = load_data()

   
    labels, kmeans, sil_score = run_clustering(X_train_pca, X_train)

 
    logreg, metrics_lr = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    rf_clf, metrics_rf = train_random_forest_classifier(
        X_train, y_train, X_test, y_test, feature_names
    )

    reg_model, metrics_reg = train_regression(None, None)

    results = {
        'LogisticRegression': metrics_lr,
        'RandomForest':       metrics_rf,
    }
    best_model, best_name = select_best_classifier(results, X_test, y_test)

    print(f"""
{'='*65}
RÉSUMÉ FINAL
{'='*65}
CLUSTERING
  Algorithme     : K-Means (k=4)
  Silhouette     : {sil_score:.4f}
  Modèle         : models/kmeans.pkl

CLASSIFICATION — LogisticRegression
  Accuracy       : {metrics_lr['accuracy']:.4f}
  F1-Score       : {metrics_lr['f1']:.4f}
  AUC-ROC        : {metrics_lr['auc_roc']:.4f}

CLASSIFICATION — RandomForest
  Accuracy       : {metrics_rf['accuracy']:.4f}
  F1-Score       : {metrics_rf['f1']:.4f}
  AUC-ROC        : {metrics_rf['auc_roc']:.4f}

Meilleur modèle  : {best_name} → models/best_model.pkl

RÉGRESSION (MonetaryTotal)
  {'R²: ' + str(round(metrics_reg['R2'], 4)) if metrics_reg else 'Non exécutée (données manquantes)'}

Visualisations   : reports/
{'='*65}
    """)


if __name__ == "__main__":
    main()
