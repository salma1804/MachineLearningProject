"""
app.py — Flask Web Application (version complète)
"""

import sys, os, traceback, joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '..'))
SRC  = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

print(f"[INFO] ROOT : {ROOT}")
print(f"[INFO] CWD  : {os.getcwd()}")

from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

# ============================================================
# CHARGEMENT MODÈLES
# ============================================================
try:
    CLF    = joblib.load('models/best_classifier.pkl')
    REG    = joblib.load('models/rf_regressor_model.pkl')
    KMEANS = joblib.load('models/kmeans_model.pkl')
    PCA    = joblib.load('models/pca_model.pkl')
    SCALER = joblib.load('models/scaler_new_features.pkl')

    CLF_FEATURES = list(CLF.feature_names_in_)   # 105 features
    REG_FEATURES = list(REG.feature_names_in_)   # 104 features
    PCA_FEATURES = list(PCA.feature_names_in_)   # 110 features (105 + 5 engineerées)

    PREDICT_OK  = True
    PREDICT_ERR = None
    print(f"[OK] CLF attend {len(CLF_FEATURES)} features")
    print(f"[OK] REG attend {len(REG_FEATURES)} features")
    print(f"[OK] PCA attend {len(PCA_FEATURES)} features")

except Exception as e:
    PREDICT_OK, PREDICT_ERR = False, str(e)
    CLF_FEATURES = REG_FEATURES = PCA_FEATURES = []
    print(f"[ERREUR] {e}")
    traceback.print_exc()


# ============================================================
# HELPERS
# ============================================================

def add_engineered_features(df):
    df = df.copy()
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
    if 'CancelledTransactions' in df.columns and 'Frequency' in df.columns:
        df['CancelRate'] = df['CancelledTransactions'] / (df['Frequency'] + 1)
    if all(c in df.columns for c in ['Frequency', 'SatisfactionScore', 'SupportTicketsCount']):
        df['EngagementScore'] = (df['Frequency'] * df['SatisfactionScore']) / (df['SupportTicketsCount'] + 1)
    return df

def scale_new(df):
    df = df.copy()
    cols = ['MonetaryPerDay', 'AvgBasketValue', 'TenureRatio', 'CancelRate', 'EngagementScore']
    present = [c for c in cols if c in df.columns]
    if present:
        df[present] = SCALER.transform(df[present])
    return df

def align(df, expected):
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            df[col] = 0.0
    return df[expected]


# ============================================================
# ROUTES PAGES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')


# ============================================================
# API ÉVALUATION
# ============================================================

@app.route('/api/evaluate')
def api_evaluate():
    try:
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            roc_auc_score, roc_curve,
            mean_squared_error, mean_absolute_error, r2_score
        )

        tt = os.path.join(ROOT, 'data', 'train_test')
        X_test      = pd.read_csv(os.path.join(tt, 'X_test.csv'))
        y_test      = pd.read_csv(os.path.join(tt, 'y_test.csv')).squeeze()
        X_test_pca  = pd.read_csv(os.path.join(tt, 'X_test_pca.csv'))
        X_train     = pd.read_csv(os.path.join(tt, 'X_train.csv'))
        y_train     = pd.read_csv(os.path.join(tt, 'y_train.csv')).squeeze()
        X_train_pca = pd.read_csv(os.path.join(tt, 'X_train_pca.csv'))

        nb_clf = joblib.load('models/naive_bayes_model.pkl')
        rf_reg = joblib.load('models/rf_regressor_model.pkl')
        ridge  = joblib.load('models/ridge_regression_model.pkl')

        X_test_clf = X_test.copy()
        for col in CLF_FEATURES:
            if col not in X_test_clf.columns:
                X_test_clf[col] = 0.0
        X_test_clf = X_test_clf[CLF_FEATURES]

        def clf_m(model, X, name):
            yp   = model.predict(X)
            prob = model.predict_proba(X)[:, 1]
            rep  = classification_report(y_test, yp, output_dict=True)
            auc  = roc_auc_score(y_test, prob)
            cm   = confusion_matrix(y_test, yp).tolist()
            fpr, tpr, _ = roc_curve(y_test, prob)
            step = max(1, len(fpr) // 100)
            return {
                'name'      : name,
                'auc'       : round(float(auc), 4),
                'accuracy'  : round(float(rep['accuracy']), 4),
                'precision' : round(float(rep['1']['precision']), 4),
                'recall'    : round(float(rep['1']['recall']), 4),
                'f1'        : round(float(rep['1']['f1-score']), 4),
                'cm'        : cm,
                'roc_fpr'   : [round(float(x), 4) for x in fpr[::step].tolist()],
                'roc_tpr'   : [round(float(x), 4) for x in tpr[::step].tolist()],
                'support_0' : int(rep['0']['support']),
                'support_1' : int(rep['1']['support']),
            }

        rf_m = clf_m(CLF,    X_test_clf, 'Random Forest')
        nb_m = clf_m(nb_clf, X_test_pca, 'Naive Bayes')

        imp = pd.Series(CLF.feature_importances_, index=CLF_FEATURES).nlargest(20).sort_values()
        feat_imp = {
            'features': imp.index.tolist(),
            'values'  : [round(float(v), 5) for v in imp.values.tolist()],
        }

        def reg_m(model, name):
            reg_feat = list(model.feature_names_in_)
            Xr = X_test.copy().drop(columns=['MonetaryTotal'], errors='ignore')
            yr = X_test['MonetaryTotal'].copy() if 'MonetaryTotal' in X_test.columns else None
            if yr is None:
                return None
            for col in reg_feat:
                if col not in Xr.columns:
                    Xr[col] = 0.0
            Xr = Xr[reg_feat]
            yp = model.predict(Xr)
            idx = np.random.choice(len(yr), min(150, len(yr)), replace=False)
            return {
                'name'        : name,
                'r2'          : round(float(r2_score(yr, yp)), 4),
                'rmse'        : round(float(np.sqrt(mean_squared_error(yr, yp))), 2),
                'mae'         : round(float(mean_absolute_error(yr, yp)), 2),
                'scatter_real': [round(float(v), 2) for v in yr.iloc[idx].tolist()],
                'scatter_pred': [round(float(v), 2) for v in yp[idx].tolist()],
            }

        rf_reg_m = reg_m(rf_reg, 'Random Forest')
        ridge_m  = reg_m(ridge,  'Ridge')

        train_clusters = KMEANS.predict(X_train_pca.values)
        cluster_dist = {}
        for c in range(KMEANS.n_clusters):
            mask = train_clusters == c
            cluster_dist[str(c)] = {
                'count'     : int(mask.sum()),
                'churn_rate': round(float(y_train[mask].mean()) if mask.sum() > 0 else 0, 3),
                'pct'       : round(float(mask.sum()) / len(train_clusters) * 100, 1),
            }

        return jsonify({
            'success': True,
            'classification': {
                'models'    : [rf_m, nb_m],
                'best'      : 'Random Forest' if rf_m['auc'] >= nb_m['auc'] else 'Naive Bayes',
                'churn_rate': round(float(y_test.mean()), 4),
                'n_test'    : int(len(y_test)),
                'n_train'   : int(len(y_train)),
            },
            'feature_importance': feat_imp,
            'regression': {
                'models': [m for m in [rf_reg_m, ridge_m] if m],
                'best'  : 'Random Forest' if (rf_reg_m and ridge_m and rf_reg_m['r2'] >= ridge_m['r2']) else 'Ridge',
            },
            'clustering': {
                'n_clusters'  : int(KMEANS.n_clusters),
                'distribution': cluster_dist,
                'inertia'     : round(float(KMEANS.inertia_), 0),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# API PRÉDICTION
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():
    if not PREDICT_OK:
        return jsonify({'success': False, 'error': f'Modèles non chargés : {PREDICT_ERR}'}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Aucune donnée reçue'}), 400

        def f(key, default=0):
            try:    return float(data.get(key, default))
            except: return float(default)
        def s(key):
            return str(data.get(key, ''))

        customer = {
            'Recency'                  : f('Recency', 30),
            'Frequency'                : f('Frequency', 5),
            'MonetaryTotal'            : f('MonetaryTotal', 500),
            'MonetaryStd'              : f('MonetaryStd', 80),
            'MonetaryMin'              : f('MonetaryMin', 20),
            'MonetaryMax'              : f('MonetaryMax', 200),
            'TotalQuantity'            : f('TotalQuantity', 100),
            'AvgQuantityPerTransaction': f('AvgQuantityPerTransaction', 20),
            'MaxQuantity'              : f('MaxQuantity', 50),
            'CustomerTenureDays'       : f('CustomerTenureDays', 365),
            'FirstPurchaseDaysAgo'     : f('FirstPurchaseDaysAgo', 365),
            'PreferredDayOfWeek'       : f('PreferredDayOfWeek', 2),
            'PreferredHour'            : f('PreferredHour', 14),
            'PreferredMonth'           : f('PreferredMonth', 6),
            'WeekendPurchaseRatio'     : f('WeekendPurchaseRatio', 0.3),
            'AvgDaysBetweenPurchases'  : f('AvgDaysBetweenPurchases', 30),
            'UniqueProducts'           : f('UniqueProducts', 10),
            'UniqueCountries'          : f('UniqueCountries', 1),
            'ZeroPriceCount'           : f('ZeroPriceCount', 0),
            'CancelledTransactions'    : f('CancelledTransactions', 0),
            'ReturnRatio'              : f('ReturnRatio', 0.05),
            'TotalTransactions'        : f('TotalTransactions', 10),
            'AvgLinesPerInvoice'       : f('AvgLinesPerInvoice', 3),
            'Age'                      : f('Age', 35),
            'SupportTicketsCount'      : f('SupportTicketsCount', 1),
            'SatisfactionScore'        : f('SatisfactionScore', 3),
            'RegYear'                  : f('RegYear', 2021),
            'RegMonth'                 : f('RegMonth', 6),
            'RegDay'                   : f('RegDay', 15),
            'RegWeekday'               : f('RegWeekday', 2),
            'SpendingCategory'         : f('SpendingCategory', 1),
            'LoyaltyLevel'             : f('LoyaltyLevel', 1),
            'ChurnRiskCategory'        : f('ChurnRiskCategory', 1),
            'AgeCategory'              : f('AgeCategory', 2),
            'BasketSizeCategory'       : f('BasketSizeCategory', 1),
        }

        for prefix, val, valid in [
            ('RFMSegment',        s('RFMSegment'),        ['Dormants','Fidèles','Potentiels']),
            ('CustomerType',      s('CustomerType'),      ['Nouveau','Occasionnel','Perdu','Régulier']),
            ('FavoriteSeason',    s('FavoriteSeason'),    ['Hiver','Printemps','Été']),
            ('PreferredTimeOfDay',s('PreferredTimeOfDay'),['Matin','Midi','Soir']),
            ('WeekendPreference', s('WeekendPreference'), ['Semaine','Weekend']),
            ('ProductDiversity',  s('ProductDiversity'),  ['Modéré','Spécialisé']),
            ('Gender',            s('Gender'),            ['M','Unknown']),
            ('AccountStatus',     s('AccountStatus'),     ['Closed','Pending','Suspended']),
            ('Region',            s('Region'),            ['Amérique du Nord','Amérique du Sud','Asie','Autre','Europe centrale','Europe continentale',"Europe de l'Est",'Europe du Nord','Europe du Sud','Moyen-Orient','Océanie','UK']),
            ('Country',           s('Country'),           ['Austria','Bahrain','Belgium','Brazil','Canada','Channel Islands','Cyprus','Czech Republic','Denmark','EIRE','European Community','Finland','France','Germany','Greece','Iceland','Israel','Italy','Japan','Lebanon','Lithuania','Malta','Netherlands','Norway','Poland','Portugal','RSA','Saudi Arabia','Singapore','Spain','Sweden','Switzerland','USA','United Arab Emirates','United Kingdom','Unspecified']),
        ]:
            if val in valid:
                customer[f'{prefix}_{val}'] = 1.0

        # ── PIPELINE ──
        # Étape 1 : DataFrame brut
        df = pd.DataFrame([customer])

        # Étape 2 : ajouter les 5 features engineerées
        df = add_engineered_features(df)

        # Étape 3 : scaler les nouvelles features
        df = scale_new(df)

        # Étape 4 : copie complète avec toutes les features disponibles
        df_full = df.copy()

        # Classification : CLF attend 105 features (sans les 5 engineerées)
        X_clf       = align(df_full, CLF_FEATURES)
        churn_pred  = int(CLF.predict(X_clf)[0])
        churn_proba = float(CLF.predict_proba(X_clf)[0][1])

        # Régression : REG attend 104 features (sans MonetaryTotal)
        df_reg   = df_full.copy().drop(columns=['MonetaryTotal'], errors='ignore')
        X_reg    = align(df_reg, REG_FEATURES)
        spending = float(REG.predict(X_reg)[0])

        # PCA : attend 110 features (105 + 5 engineerées) → PCA_FEATURES
        X_for_pca = align(df_full, PCA_FEATURES)
        X_pca     = PCA.transform(X_for_pca)
        cluster   = int(KMEANS.predict(X_pca)[0])

        return jsonify({
            'success'           : True,
            'churn_prediction'  : churn_pred,
            'churn_label'       : 'Churn' if churn_pred == 1 else 'Fidèle',
            'churn_probability' : round(churn_proba, 4),
            'predicted_spending': round(spending, 2),
            'cluster'           : cluster,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# HEALTH
# ============================================================

@app.route('/health')
def health():
    return jsonify({
        'status'      : 'ok' if PREDICT_OK else 'degraded',
        'clf_features': len(CLF_FEATURES),
        'reg_features': len(REG_FEATURES),
        'pca_features': len(PCA_FEATURES),
        'cwd'         : os.getcwd(),
    })


if __name__ == '__main__':
    print("=" * 55)
    print("  ML Dashboard — Analyse Comportementale Retail")
    print("=" * 55)
    print("  Prédiction  : http://127.0.0.1:5000")
    print("  Évaluation  : http://127.0.0.1:5000/evaluation")
    print("  Health      : http://127.0.0.1:5000/health")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)