import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import os
import warnings
warnings.filterwarnings('ignore')



def load_data(data_path):
    """Charge les données depuis un fichier CSV."""
    return pd.read_csv(data_path)


def save_model(model, model_path):
    """Sauvegarde un modèle entraîné avec joblib."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé : {model_path}")


def load_model(model_path):
    """Charge un modèle sauvegardé avec joblib."""
    return joblib.load(model_path)



def explore_dataframe(df, name="Dataset"):
    """
    Affiche un résumé complet du DataFrame :
    shape, types, valeurs manquantes, statistiques descriptives.
    """
    print(f"\n{'='*60}")
    print(f"EXPLORATION : {name}")
    print(f"{'='*60}")
    print(f"Shape         : {df.shape}")
    print(f"Doublons      : {df.duplicated().sum()}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\nValeurs manquantes :")
        for col, n in missing.items():
            print(f"  {col:35s} {n:5d}  ({n/len(df)*100:.1f}%)")
    else:
        print("\nAucune valeur manquante.")
    print(f"\nTypes :\n{df.dtypes.value_counts().to_string()}")
    print(f"\nStatistiques numériques :")
    print(df.describe().round(2).to_string())


def plot_missing_values(df, save_path=None):
    """Heatmap des valeurs manquantes."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        print("Aucune valeur manquante.")
        return
    plt.figure(figsize=(10, max(4, len(missing_pct) * 0.4)))
    sns.barplot(x=missing_pct.values, y=missing_pct.index, palette='Reds_r')
    plt.title('Taux de valeurs manquantes par feature')
    plt.xlabel('% manquant')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé : {save_path}")
    plt.show()


def plot_correlation_heatmap(df, save_path=None, threshold=0.8):
    """
    Heatmap de corrélation sur les features numériques.
    Affiche aussi les paires fortement corrélées (> threshold).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Paires fortement corrélées
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) >= threshold:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    if high_corr:
        print(f"\nPaires corrélées (|r| >= {threshold}) :")
        for a, b, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {a:35s} <-> {b:35s}  r={r:.3f}")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm',
                center=0, linewidths=0.3, vmin=-1, vmax=1)
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap sauvegardée : {save_path}")
    plt.show()
    return corr


def plot_target_distribution(y, target_name='Churn', save_path=None):
    """Distribution de la variable cible."""
    counts = pd.Series(y).value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette='Blues_d')
    for i, v in enumerate(counts.values):
        plt.text(i, v + 5, f"{v}\n({v/len(y)*100:.1f}%)", ha='center', fontsize=10)
    plt.title(f'Distribution de {target_name}')
    plt.xlabel(target_name)
    plt.ylabel('Nombre de clients')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def run_pca(X, n_components=None, variance_threshold=0.95, save_dir='reports'):
    
    os.makedirs(save_dir, exist_ok=True)

   
    pca_full = PCA(random_state=42)
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    if n_components is None:
        n_components = int(np.argmax(cumvar >= variance_threshold) + 1)
    print(f"\nACP : {n_components} composantes retenues "
          f"({cumvar[n_components-1]*100:.1f}% de variance expliquée)")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, min(21, len(pca_full.explained_variance_ratio_)+1)),
            pca_full.explained_variance_ratio_[:20] * 100, color='steelblue')
    plt.xlabel('Composante')
    plt.ylabel('Variance expliquée (%)')
    plt.title('Variance par composante (top 20)')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumvar)+1), cumvar * 100, marker='o', markersize=3,
             color='steelblue')
    plt.axhline(y=variance_threshold*100, color='red', linestyle='--',
                label=f'{variance_threshold*100:.0f}%')
    plt.axvline(x=n_components, color='orange', linestyle='--',
                label=f'n={n_components}')
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Variance cumulée (%)')
    plt.title('Variance cumulée')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pca_variance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Graphique ACP sauvegardé : {save_dir}/pca_variance.png")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    plot_pca_2d(X_pca, save_dir=save_dir)

    return X_pca, pca


def plot_pca_2d(X_pca, labels=None, label_name='Cluster', save_dir='reports'):
    """Scatter plot des 2 premières composantes principales."""
    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=labels, cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter, label=label_name)
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, s=10, color='steelblue')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Projection ACP — 2 premières composantes')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pca_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Graphique ACP 2D sauvegardé : {save_dir}/pca_2d.png")


def plot_pca_loadings(pca, feature_names, top_n=10, save_dir='reports'):
    """
    Affiche les loadings (contributions) des features sur PC1 et PC2.
    """
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=feature_names,
        columns=['PC1', 'PC2']
    )
    loadings['magnitude'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
    top = loadings.nlargest(top_n, 'magnitude')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, comp in zip(axes, ['PC1', 'PC2']):
        data = top[comp].sort_values()
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in data.values]
        ax.barh(data.index, data.values, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f'Loadings — {comp}')
        ax.set_xlabel('Contribution')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pca_loadings.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Loadings ACP sauvegardés : {save_dir}/pca_loadings.png")
    return loadings




def find_optimal_k(X_pca, k_range=range(2, 11), save_dir='reports'):
   
    os.makedirs(save_dir, exist_ok=True)
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        inertias[k] = km.inertia_

    plt.figure(figsize=(8, 5))
    plt.plot(list(inertias.keys()), list(inertias.values()),
             marker='o', color='steelblue', linewidth=2)
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude — choix de k')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/kmeans_elbow.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Graphique coude sauvegardé : {save_dir}/kmeans_elbow.png")
    return inertias


def run_kmeans(X_pca, n_clusters=4, save_dir='reports'):
    """
    Entraîne K-Means et retourne les labels + l'objet KMeans.

    Returns
    -------
    labels : np.ndarray, kmeans : KMeans fitted
    """
    os.makedirs(save_dir, exist_ok=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    print(f"\nClustering K-Means (k={n_clusters})")
    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"  Cluster {c} : {n} clients ({n/len(labels)*100:.1f}%)")

   
    plot_pca_2d(X_pca, labels=labels, label_name='Cluster', save_dir=save_dir)

    return labels, kmeans


def describe_clusters(df_original, labels, rfm_cols=None, save_dir='reports'):
    """
    Profil moyen de chaque cluster sur les features RFM + comportementales.

    Parameters
    ----------
    df_original : DataFrame original (non normalisé, features numériques)
    labels      : array de labels de cluster
    rfm_cols    : liste de colonnes à analyser (défaut : Recency, Frequency, MonetaryTotal)
    """
    os.makedirs(save_dir, exist_ok=True)
    if rfm_cols is None:
        rfm_cols = [c for c in ['Recency', 'Frequency', 'MonetaryTotal',
                                 'CustomerTenure', 'UniqueProducts', 'ReturnRatio']
                    if c in df_original.columns]

    df_c = df_original[rfm_cols].copy()
    df_c['Cluster'] = labels
    profile = df_c.groupby('Cluster')[rfm_cols].mean().round(2)

    print("\nProfil moyen par cluster :")
    print(profile.to_string())

    # Heatmap profils
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    plt.figure(figsize=(max(8, len(rfm_cols)), max(4, len(profile))))
    sns.heatmap(profile_norm, annot=profile.values, fmt='.1f',
                cmap='YlOrRd', linewidths=0.3)
    plt.title('Profil normalisé des clusters')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cluster_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Profils clusters sauvegardés : {save_dir}/cluster_profiles.png")
    return profile



def plot_confusion_matrix(y_true, y_pred, model_name='Modèle', save_path=None):
    """Matrice de confusion annotée."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fidèle', 'Churn'],
                yticklabels=['Fidèle', 'Churn'])
    plt.title(f'Matrice de confusion — {model_name}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Matrice sauvegardée : {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_proba, model_name='Modèle', save_path=None):
    """Courbe ROC avec AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2, color='steelblue')
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    plt.xlabel('Taux de faux positifs (FPR)')
    plt.ylabel('Taux de vrais positifs (TPR)')
    plt.title(f'Courbe ROC — {model_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Courbe ROC sauvegardée : {save_path}")
    plt.show()
    return auc


def evaluate_classifier(model, X_test, y_test, model_name='Modèle'):
    """
    Évalue complètement un modèle de classification.
    Retourne un dict de métriques.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'auc_roc':   roc_auc_score(y_test, y_proba)
    }

    print(f"\n{'='*55}")
    print(f"ÉVALUATION CLASSIFICATION : {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Fidèle','Churn'])}")

    if metrics['f1'] > 0.99:
        print("  ⚠  Score parfait → vérifier data leakage !")

    return metrics, y_pred, y_proba



def evaluate_regressor(model, X_test, y_test, model_name='Modèle'):
    """
    Évalue un modèle de régression.
    Retourne un dict de métriques.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'MAE':  mean_absolute_error(y_test, y_pred),
        'MSE':  mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2':   r2_score(y_test, y_pred)
    }

    print(f"\n{'='*55}")
    print(f"ÉVALUATION RÉGRESSION : {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:6s}: {v:.4f}")

    # Résidus
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.4, s=8, color='steelblue')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('Valeur réelle')
    plt.ylabel('Valeur prédite')
    plt.title(f'Réel vs Prédit — {model_name}')

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=40, color='steelblue', edgecolor='white')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Résidu')
    plt.ylabel('Fréquence')
    plt.title('Distribution des résidus')
    plt.tight_layout()
    plt.show()

    return metrics, y_pred


def plot_feature_importance(model, feature_names, top_n=15,
                             model_name='Modèle', save_path=None):
    """
    Barplot des features les plus importantes.
    Compatible avec tree-based models (feature_importances_)
    et modèles linéaires (coef_).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        print("Modèle non compatible.")
        return None

    imp_df = pd.DataFrame({
        'feature':    feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, max(5, top_n * 0.4)))
    sns.barplot(data=imp_df, x='importance', y='feature', palette='Blues_r')
    plt.title(f'Top {top_n} features importantes — {model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Feature importance sauvegardée : {save_path}")
    plt.show()
    return imp_df



def detect_overfitting(model, X_train, y_train, X_test, y_test):
    """
    Compare les scores train / test pour détecter le surapprentissage.
    Fonctionne pour classification et régression.
    """
    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test, y_test)
    diff = train_score - test_score

    print(f"\n  Score Train : {train_score:.4f}")
    print(f"  Score Test  : {test_score:.4f}")
    print(f"  Différence  : {diff:.4f}", end='  ')

    if diff > 0.10:
        print("⚠  Surapprentissage détecté !")
    elif diff < 0.02:
        print("✓  Excellent équilibre")
    else:
        print("ℹ  Légère divergence, acceptable")

    return train_score, test_score


def compare_models(results_dict, metric='f1', save_path=None):
    """
    Affiche un barplot comparatif de plusieurs modèles.

    Parameters
    ----------
    results_dict : {nom_modèle: dict_métriques}
    metric       : str — métrique à afficher ('f1', 'auc_roc', 'accuracy', 'R2'…)
    """
    names  = list(results_dict.keys())
    values = [results_dict[n].get(metric, 0) for n in names]

    plt.figure(figsize=(max(6, len(names) * 1.5), 5))
    bars = plt.bar(names, values, color='steelblue', edgecolor='white')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.ylim(0, min(1.05, max(values) * 1.15))
    plt.ylabel(metric)
    plt.title(f'Comparaison des modèles — {metric}')
    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Comparaison sauvegardée : {save_path}")
    plt.show()
