"""
=====================================================================
utils.py — Fonctions Utilitaires Partagées
Projet : Analyse Comportementale Clientèle Retail
Auteur : GI2 — Atelier Machine Learning

Ce module regroupe toutes les fonctions réutilisables du projet :
  - Chargement et sauvegarde de modèles
  - Visualisation (courbes ROC, matrices, importances)
  - Imputation et nettoyage
  - Feature engineering
  - Métriques d'évaluation
  - Journalisation (logging)
=====================================================================
"""

import os
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION DU LOGGING
# ============================================================

def setup_logger(name: str = "ml_project", log_file: str = None) -> logging.Logger:
    """
    Initialise et retourne un logger configuré.

    Args:
        name     : Nom du logger (par défaut 'ml_project')
        log_file : Chemin vers un fichier log (facultatif)

    Returns:
        logging.Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler console
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler fichier (optionnel)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# ============================================================
# GESTION DES FICHIERS ET MODÈLES
# ============================================================

def ensure_dirs(*dirs: str) -> None:
    """
    Crée les dossiers s'ils n'existent pas encore.

    Args:
        *dirs : Chemins des dossiers à créer
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Dossier vérifié/créé : {d}")


def save_model(model, path: str) -> None:
    """
    Sauvegarde un modèle scikit-learn avec joblib.

    Args:
        model : Modèle entraîné à sauvegarder
        path  : Chemin de destination (.pkl)
    """
    ensure_dirs(os.path.dirname(path))
    joblib.dump(model, path)
    logger.info(f"Modèle sauvegardé : {path}")


def load_model(path: str):
    """
    Charge un modèle depuis un fichier .pkl.

    Args:
        path : Chemin du modèle sauvegardé

    Returns:
        Modèle chargé

    Raises:
        FileNotFoundError si le fichier est absent
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    model = joblib.load(path)
    logger.info(f"Modèle chargé : {path}")
    return model


def load_train_test(base_dir: str = 'data/train_test'):
    """
    Charge X_train, X_test, y_train, y_test depuis le dossier train_test.

    Args:
        base_dir : Chemin du dossier contenant les CSV

    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    X_train = pd.read_csv(f'{base_dir}/X_train.csv')
    X_test  = pd.read_csv(f'{base_dir}/X_test.csv')
    y_train = pd.read_csv(f'{base_dir}/y_train.csv').squeeze()
    y_test  = pd.read_csv(f'{base_dir}/y_test.csv').squeeze()

    logger.info(f"Train chargé : X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test  chargé : X={X_test.shape},  y={y_test.shape}")
    return X_train, X_test, y_train, y_test


def load_pca_data(base_dir: str = 'data/train_test'):
    """
    Charge les données transformées par PCA.

    Returns:
        Tuple (X_train_pca, X_test_pca)
    """
    X_train_pca = pd.read_csv(f'{base_dir}/X_train_pca.csv')
    X_test_pca  = pd.read_csv(f'{base_dir}/X_test_pca.csv')
    logger.info(f"PCA chargé : train={X_train_pca.shape}, test={X_test_pca.shape}")
    return X_train_pca, X_test_pca


# ============================================================
# NETTOYAGE ET IMPUTATION
# ============================================================

def cap_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Limite les valeurs aberrantes aux bornes IQR (méthode de Tukey).
    Valeurs < Q1 - 1.5*IQR  →  remplacées par la borne inférieure
    Valeurs > Q3 + 1.5*IQR  →  remplacées par la borne supérieure

    Args:
        series : Colonne numérique pandas

    Returns:
        Série avec outliers plafonnés
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)


def impute_with_median(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Impute les NaN d'une colonne numérique par la médiane.

    Args:
        df  : DataFrame source
        col : Nom de la colonne à imputer

    Returns:
        DataFrame modifié (in-place)
    """
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    logger.info(f"Imputation médiane '{col}' : {median_val:.4f}")
    return df


def impute_with_mode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Impute les NaN d'une colonne catégorielle par le mode.

    Args:
        df  : DataFrame source
        col : Nom de la colonne à imputer

    Returns:
        DataFrame modifié (in-place)
    """
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    logger.info(f"Imputation mode '{col}' : {mode_val}")
    return df


def replace_and_impute(df: pd.DataFrame, col: str,
                        invalid_values: list, method: str = 'median') -> pd.DataFrame:
    """
    Remplace des valeurs invalides par NaN puis impute.

    Args:
        df             : DataFrame source
        col            : Colonne à traiter
        invalid_values : Liste des valeurs à considérer comme NaN (ex: [-1, 999])
        method         : 'median' ou 'mode'

    Returns:
        DataFrame modifié
    """
    df[col] = df[col].replace(invalid_values, np.nan)
    if method == 'median':
        return impute_with_median(df, col)
    elif method == 'mode':
        return impute_with_mode(df, col)
    else:
        raise ValueError(f"Méthode inconnue : {method}. Utilisez 'median' ou 'mode'.")


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes avec une seule valeur unique (variance nulle).

    Args:
        df : DataFrame source

    Returns:
        DataFrame sans les colonnes constantes
    """
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        logger.info(f"Colonnes constantes supprimées : {constant_cols}")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles features comportementales à partir des features existantes.

    Nouvelles features créées :
      - MonetaryPerDay   : Dépense journalière moyenne (MonetaryTotal / Recency+1)
      - AvgBasketValue   : Panier moyen par commande   (MonetaryTotal / Frequency+1)
      - TenureRatio      : Ratio ancienneté / récence  (Recency / CustomerTenureDays+1)
      - CancelRate       : Taux d'annulation           (CancelledTransactions / Frequency+1)
      - EngagementScore  : Score d'engagement global   (Frequency * Satisfaction / Tickets+1)

    Args:
        df : DataFrame (X_train ou X_test)

    Returns:
        DataFrame enrichi avec les nouvelles features
    """
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)

    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)

    if 'CancelledTransactions' in df.columns and 'Frequency' in df.columns:
        df['CancelRate'] = df['CancelledTransactions'] / (df['Frequency'] + 1)

    if all(c in df.columns for c in ['Frequency', 'SatisfactionScore', 'SupportTicketsCount']):
        df['EngagementScore'] = (
            df['Frequency'] * df['SatisfactionScore']
        ) / (df['SupportTicketsCount'] + 1)

    return df


def scale_new_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        new_features: list, save_path: str = None):
    """
    Applique StandardScaler sur les nouvelles features engineerées.
    Fit uniquement sur X_train pour éviter le data leakage.

    Args:
        X_train      : Données d'entraînement
        X_test       : Données de test
        new_features : Liste des features à scaler
        save_path    : Chemin pour sauvegarder le scaler (facultatif)

    Returns:
        Tuple (X_train, X_test, scaler)
    """
    # Ne garder que les features effectivement présentes
    features = [f for f in new_features if f in X_train.columns]

    scaler = StandardScaler()
    X_train[features] = scaler.fit_transform(X_train[features])
    X_test[features]  = scaler.transform(X_test[features])

    if save_path:
        save_model(scaler, save_path)

    logger.info(f"Scaling appliqué sur : {features}")
    return X_train, X_test, scaler


# ============================================================
# VISUALISATION — CLASSIFICATION
# ============================================================

def plot_confusion_matrix(y_true, y_pred, model_name: str,
                           save_path: str = None,
                           labels: list = None) -> None:
    """
    Trace et sauvegarde la matrice de confusion.

    Args:
        y_true     : Labels réels
        y_pred     : Labels prédits
        model_name : Nom du modèle (pour le titre)
        save_path  : Chemin de sauvegarde du graphique
        labels     : Liste des noms de classes (ex: ['Fidèle', 'Churn'])
    """
    if labels is None:
        labels = ['Classe 0', 'Classe 1']

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Matrice de Confusion — {model_name}')
    plt.tight_layout()

    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=120)
        logger.info(f"Matrice de confusion sauvegardée : {save_path}")
    plt.close()


def plot_roc_curves(models_data: list, y_test,
                    save_path: str = 'reports/roc_curves.png') -> None:
    """
    Trace les courbes ROC pour plusieurs modèles sur un même graphique.

    Args:
        models_data : Liste de dicts avec clés 'name', 'proba' (array probabilités)
        y_test      : Labels réels
        save_path   : Chemin de sauvegarde
    
    Exemple d'utilisation :
        plot_roc_curves([
            {'name': 'Random Forest', 'proba': rf_proba},
            {'name': 'Naive Bayes',   'proba': nb_proba},
        ], y_test)
    """
    colors = ['steelblue', 'tomato', 'mediumseagreen', 'darkorange', 'purple']

    plt.figure(figsize=(8, 6))

    for i, m in enumerate(models_data):
        fpr, tpr, _ = roc_curve(y_test, m['proba'])
        auc = roc_auc_score(y_test, m['proba'])
        plt.plot(fpr, tpr,
                 color=colors[i % len(colors)],
                 label=f"{m['name']} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')
    plt.xlabel('Taux Faux Positifs (FPR)')
    plt.ylabel('Taux Vrais Positifs (TPR)')
    plt.title('Courbes ROC — Comparaison des modèles')
    plt.legend(loc='lower right')
    plt.tight_layout()

    ensure_dirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info(f"Courbes ROC sauvegardées : {save_path}")


def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                             title: str = 'Feature Importance',
                             save_path: str = None) -> None:
    """
    Trace un graphique en barres horizontales pour l'importance des features.
    Compatible avec RandomForest, GradientBoosting, XGBoost, etc.

    Args:
        model         : Modèle entraîné avec attribut feature_importances_
        feature_names : Noms des features
        top_n         : Nombre de features à afficher
        title         : Titre du graphique
        save_path     : Chemin de sauvegarde
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Ce modèle n'a pas d'attribut feature_importances_.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    plt.figure(figsize=(10, 7))
    top.plot(kind='barh', color='steelblue')
    plt.title(title)
    plt.xlabel('Importance (Gini)')
    plt.tight_layout()

    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=120)
        logger.info(f"Importance features sauvegardée : {save_path}")
    plt.close()


# ============================================================
# VISUALISATION — RÉGRESSION
# ============================================================

def plot_regression_results(y_true, y_pred, model_name: str,
                             target_name: str = 'Target',
                             save_path: str = None) -> None:
    """
    Trace le graphique Valeurs réelles vs Valeurs prédites.

    Args:
        y_true      : Valeurs réelles
        y_pred      : Valeurs prédites
        model_name  : Nom du modèle
        target_name : Nom de la variable cible (pour les axes)
        save_path   : Chemin de sauvegarde
    """
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    mn = min(float(y_true.min()), float(y_pred.min()))
    mx = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([mn, mx], [mn, mx], 'r--', label='Prédiction parfaite')
    plt.xlabel(f'Valeurs réelles ({target_name})')
    plt.ylabel('Valeurs prédites')
    plt.title(f'{model_name} — Réel vs Prédit\nR²={r2:.3f}  RMSE={rmse:.2f}')
    plt.legend()
    plt.tight_layout()

    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=120)
        logger.info(f"Graphique régression sauvegardé : {save_path}")
    plt.close()


# ============================================================
# MÉTRIQUES D'ÉVALUATION
# ============================================================

def evaluate_classifier(name: str, model, X_train, X_test,
                         y_train, y_test,
                         save_dir: str = 'reports') -> dict:
    """
    Évalue complètement un modèle de classification.
    Calcule : Accuracy, Precision, Recall, F1, ROC-AUC, CV-AUC.
    Génère et sauvegarde la matrice de confusion.

    Args:
        name     : Nom du modèle
        model    : Modèle entraîné
        X_train  : Features d'entraînement
        X_test   : Features de test
        y_train  : Labels d'entraînement
        y_test   : Labels de test
        save_dir : Répertoire pour les graphiques

    Returns:
        dict avec toutes les métriques
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    report  = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Cross-validation ROC-AUC sur train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    logger.info(f"\n{'='*50}\n  {name}\n{'='*50}")
    logger.info(f"ROC-AUC    : {roc_auc:.4f}")
    logger.info(f"CV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Precision (classe 1) : {report['1']['precision']:.4f}")
    logger.info(f"Recall    (classe 1) : {report['1']['recall']:.4f}")
    logger.info(f"F1-score  (classe 1) : {report['1']['f1-score']:.4f}")

    # Matrice de confusion
    cm_path = os.path.join(save_dir, f'confusion_{name.replace(" ", "_")}.png')
    plot_confusion_matrix(y_test, y_pred, name,
                          save_path=cm_path,
                          labels=['Fidèle', 'Churn'])

    return {
        'name'        : name,
        'roc_auc'     : roc_auc,
        'cv_mean'     : cv_scores.mean(),
        'cv_std'      : cv_scores.std(),
        'precision_1' : report['1']['precision'],
        'recall_1'    : report['1']['recall'],
        'f1_1'        : report['1']['f1-score'],
        'accuracy'    : report['accuracy']
    }


def evaluate_regressor(name: str, model, X_train, X_test,
                        y_train, y_test,
                        target_name: str = 'Target',
                        save_dir: str = 'reports') -> dict:
    """
    Évalue complètement un modèle de régression.
    Calcule : R², RMSE, MAE, CV-R².
    Génère et sauvegarde le graphique réel vs prédit.

    Args:
        name        : Nom du modèle
        model       : Modèle entraîné
        X_train     : Features d'entraînement
        X_test      : Features de test
        y_train     : Cible d'entraînement
        y_test      : Cible de test
        target_name : Nom de la variable cible
        save_dir    : Répertoire pour les graphiques

    Returns:
        dict avec toutes les métriques
    """
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    logger.info(f"\n{'='*50}\n  {name}\n{'='*50}")
    logger.info(f"R²   (test)  : {r2:.4f}")
    logger.info(f"RMSE (test)  : {rmse:.4f}")
    logger.info(f"MAE  (test)  : {mae:.4f}")
    logger.info(f"CV R²        : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # Graphique réel vs prédit
    plot_path = os.path.join(save_dir, f'regression_{name.replace(" ", "_")}.png')
    plot_regression_results(y_test, y_pred, name,
                            target_name=target_name,
                            save_path=plot_path)

    return {
        'name'   : name,
        'r2'     : r2,
        'rmse'   : rmse,
        'mae'    : mae,
        'cv_r2'  : cv_r2.mean(),
        'cv_std' : cv_r2.std()
    }


def print_summary_table(results: list, title: str = "Résumé comparatif") -> pd.DataFrame:
    """
    Affiche un tableau récapitulatif des métriques de plusieurs modèles.

    Args:
        results : Liste de dicts retournés par evaluate_classifier/evaluate_regressor
        title   : Titre du tableau

    Returns:
        DataFrame du résumé
    """
    df_summary = pd.DataFrame(results).set_index('name')
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(df_summary.round(4).to_string())
    return df_summary


# ============================================================
# VÉRIFICATIONS ET DIAGNOSTICS
# ============================================================

def check_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame) -> bool:
    """
    Vérifie si des lignes de X_test sont présentes dans X_train (data leakage).

    Args:
        X_train : DataFrame d'entraînement
        X_test  : DataFrame de test

    Returns:
        True si aucune fuite détectée, False sinon
    """
    overlap = pd.merge(X_train, X_test, how='inner')
    if len(overlap) > 0:
        logger.warning(f"⚠️  DATA LEAKAGE DÉTECTÉ : {len(overlap)} lignes communes !")
        return False
    logger.info("✓ Aucune fuite de données (data leakage) détectée.")
    return True


def check_class_balance(y: pd.Series, label: str = "Target") -> float:
    """
    Affiche la distribution des classes et retourne le taux de la classe minoritaire.

    Args:
        y     : Série cible binaire
        label : Nom de la variable

    Returns:
        Taux de la classe positive (classe 1)
    """
    counts = y.value_counts(normalize=True).round(4)
    logger.info(f"\nDistribution de {label} :")
    for cls, pct in counts.items():
        logger.info(f"  Classe {cls} : {pct:.2%}")

    minority_rate = float(counts.get(1, counts.iloc[-1]))
    if minority_rate < 0.3:
        logger.warning(f"⚠️  Classes déséquilibrées (minorité = {minority_rate:.2%}) — utilisez class_weight='balanced'")
    else:
        logger.info("✓ Classes équilibrées.")
    return minority_rate


def get_high_correlation_pairs(df: pd.DataFrame,
                                threshold: float = 0.85) -> pd.DataFrame:
    """
    Identifie les paires de variables numériques fortement corrélées.

    Args:
        df        : DataFrame numérique
        threshold : Seuil de corrélation (défaut 0.85)

    Returns:
        DataFrame avec les paires [feature_1, feature_2, correlation]
    """
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
    pairs = []
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr_matrix.iloc[i, j] > threshold:
                pairs.append({
                    'feature_1'   : cols[i],
                    'feature_2'   : cols[j],
                    'correlation' : round(corr_matrix.iloc[i, j], 4)
                })

    result = pd.DataFrame(pairs).sort_values('correlation', ascending=False)
    logger.info(f"Paires corrélées (|r| > {threshold}) : {len(result)} trouvées")
    return result


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un tableau récapitulatif des valeurs manquantes.

    Args:
        df : DataFrame à analyser

    Returns:
        DataFrame [colonne, nb_manquants, pct_manquants] pour les colonnes avec NaN
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        logger.info("✓ Aucune valeur manquante dans le DataFrame.")
        return pd.DataFrame()

    summary = pd.DataFrame({
        'nb_manquants'  : missing,
        'pct_manquants' : (missing / len(df) * 100).round(2)
    }).sort_values('pct_manquants', ascending=False)

    logger.info(f"Colonnes avec valeurs manquantes :\n{summary.to_string()}")
    return summary