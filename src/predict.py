import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

from utils import (
    load_model, logger, ensure_dirs,
    add_engineered_features
)


PATHS = {
    'best_classifier'    : 'models/best_classifier.pkl',
    'random_forest_clf'  : 'models/random_forest_model.pkl',
    'naive_bayes'        : 'models/naive_bayes_model.pkl',
    'best_regressor'     : 'models/rf_regressor_model.pkl',
    'ridge'              : 'models/ridge_regression_model.pkl',
    'rf_regressor'       : 'models/rf_regressor_model.pkl',
    'kmeans'             : 'models/kmeans_model.pkl',
    'pca'                : 'models/pca_model.pkl',
    'pca_2d'             : 'models/pca_2d_model.pkl',
    'scaler_new_features': 'models/scaler_new_features.pkl',
}

NEW_FEATURES = [
    'MonetaryPerDay', 'AvgBasketValue', 'TenureRatio',
    'CancelRate', 'EngagementScore'
]


class ModelRegistry:
    def __init__(self):
        self._cache = {}

    def get(self, key: str):
        if key not in self._cache:
            path = PATHS.get(key)
            if path is None:
                raise KeyError(f"Clé inconnue : '{key}'")
            self._cache[key] = load_model(path)
        return self._cache[key]


registry = ModelRegistry()


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df = add_engineered_features(df)
    scaler_new = registry.get('scaler_new_features')
    features_present = [f for f in NEW_FEATURES if f in df.columns]
    if features_present:
        df[features_present] = scaler_new.transform(df[features_present])
        logger.info(f"Nouvelles features scalées : {features_present}")

    return df 


def align_to(df: pd.DataFrame, expected: list) -> pd.DataFrame:
    
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            df[col] = 0.0
    return df[expected]


def apply_pca(df_full: pd.DataFrame) -> np.ndarray:
    """
    Applique la PCA sur le DataFrame COMPLET (après prepare_input).
    Utilise pca.feature_names_in_ pour aligner correctement sur les
    110 features (105 originales + 5 engineerées).
    """
    pca = registry.get('pca')
    if hasattr(pca, 'feature_names_in_'):
        pca_features = list(pca.feature_names_in_)
        df_aligned = align_to(df_full, pca_features)
    else:
        df_aligned = df_full.copy()
    return pca.transform(df_aligned)


def predict_churn(df: pd.DataFrame, use_pca: bool = False,
                  return_proba: bool = True) -> pd.DataFrame:
    logger.info("=== Prédiction CHURN ===")

    df_full = prepare_input(df)

    if use_pca:
        X = apply_pca(df_full)
        model = registry.get('naive_bayes')
        model_name = "Naive Bayes (PCA)"
    else:
        model = registry.get('best_classifier')
        model_name = "Random Forest"
        
        clf_features = list(model.feature_names_in_)
        X = align_to(df_full, clf_features).values

    logger.info(f"Modèle : {model_name} | Shape : {X.shape}")

    predictions = model.predict(X)
    result = df[[]].copy()
    result['churn_prediction'] = predictions
    result['churn_label'] = result['churn_prediction'].map({0: 'Fidèle', 1: 'Churn'})

    if return_proba:
        probas = model.predict_proba(X)[:, 1]
        result['churn_probability'] = probas.round(4)
        logger.info(f"Taux churn prédit : {predictions.mean():.2%}")

    return result


def predict_spending(df: pd.DataFrame, use_ridge: bool = False) -> pd.DataFrame:
    logger.info("=== Prédiction MONTANT DÉPENSÉ ===")

    df_full = prepare_input(df)

    if use_ridge:
        model = registry.get('ridge')
        model_name = "Ridge Regression"
    else:
        model = registry.get('best_regressor')
        model_name = "Random Forest Regressor"

  
    reg_features = list(model.feature_names_in_)
    df_reg = df_full.drop(columns=['MonetaryTotal'], errors='ignore')
    X = align_to(df_reg, reg_features).values

    logger.info(f"Modèle : {model_name} | Shape : {X.shape}")

    predictions = model.predict(X)
    result = df[[]].copy()
    result['predicted_spending'] = predictions.round(2)

    logger.info(f"Dépense moy : £{predictions.mean():.2f}")
    return result


def predict_cluster(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== Prédiction CLUSTER ===")

  
    df_full = prepare_input(df)
    X_pca = apply_pca(df_full)

    kmeans = registry.get('kmeans')
    clusters = kmeans.predict(X_pca)

    result = df[[]].copy()
    result['cluster'] = clusters

    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    logger.info(f"Distribution clusters :\n{cluster_counts.to_string()}")

    return result



def predict_full(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== PRÉDICTION COMPLÈTE ===")

    churn_results    = predict_churn(df, use_pca=False, return_proba=True)
    spending_results = predict_spending(df, use_ridge=False)
    cluster_results  = predict_cluster(df)

    result = pd.concat([churn_results, spending_results, cluster_results], axis=1)

    logger.info(f"Clients analysés     : {len(result)}")
    logger.info(f"Churners détectés    : {result['churn_prediction'].sum()}")
    logger.info(f"Dépense moy. prédite : £{result['predicted_spending'].mean():.2f}")
    logger.info(f"Segments présents    : {sorted(result['cluster'].unique())}")

    return result


def batch_predict(input_path: str, output_path: str = None,
                  mode: str = 'full') -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    logger.info(f"Chargement : {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Données : {df.shape[0]} clients, {df.shape[1]} features")

    mode = mode.lower()
    if   mode == 'churn'   : result = predict_churn(df)
    elif mode == 'spending' : result = predict_spending(df)
    elif mode == 'cluster'  : result = predict_cluster(df)
    elif mode == 'full'     : result = predict_full(df)
    else: raise ValueError(f"Mode invalide : '{mode}'")

    if 'CustomerID' in df.columns:
        result.insert(0, 'CustomerID', df['CustomerID'].values)

    if output_path is None:
        ensure_dirs('reports')
        output_path = f'reports/predictions_{mode}.csv'

    ensure_dirs(os.path.dirname(output_path))
    result.to_csv(output_path, index=False)
    logger.info(f"Sauvegardé : {output_path}")

    return result

def predict_single_customer(customer_data: dict, mode: str = 'full') -> dict:
    df = pd.DataFrame([customer_data])

    mode = mode.lower()
    if   mode == 'churn'   : result_df = predict_churn(df, return_proba=True)
    elif mode == 'spending' : result_df = predict_spending(df)
    elif mode == 'cluster'  : result_df = predict_cluster(df)
    elif mode == 'full'     : result_df = predict_full(df)
    else: raise ValueError(f"Mode invalide : '{mode}'")

    result = result_df.iloc[0].to_dict()
    logger.info(f"Prédiction ({mode}) : {result}")
    return result



def parse_args():
    parser = argparse.ArgumentParser(description='Prédiction comportementale retail')
    parser.add_argument('--mode',   type=str, default='full', choices=['churn','spending','cluster','full'])
    parser.add_argument('--input',  type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.info(f"Mode : {args.mode} | Input : {args.input}")

    try:
        results = batch_predict(input_path=args.input, output_path=args.output, mode=args.mode)
        print(f"\nClients traités : {len(results)}")
        print(results.head(10).to_string(index=False))

    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Erreur : {e}", exc_info=True)
        raise
