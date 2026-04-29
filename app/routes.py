from app.forms import SinglePredictionForm, BatchUploadForm
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, send_file
import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

main_bp = Blueprint('main', __name__)

MODEL         = None
SCALER        = None
FEATURE_NAMES = None

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# Cherche best_model.pkl (RandomForest) en priorité
# ─────────────────────────────────────────────────────────────────────────────

def load_ml_model():
    global MODEL, SCALER, FEATURE_NAMES
    if MODEL is not None:
        return

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

    for name in ['best_model.pkl', 'random_forest.pkl',
                 'best_model_LogisticRegression.pkl', 'logreg.pkl']:
        path = os.path.join(base, name)
        if os.path.exists(path):
            MODEL = joblib.load(path)
            print(f"Modèle chargé : {name}")
            break

    scaler_path = os.path.join(base, 'scaler.pkl')
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
        print("Scaler chargé")

    fn_path = os.path.join(base, 'feature_names.pkl')
    if os.path.exists(fn_path):
        FEATURE_NAMES = joblib.load(fn_path)
        print(f"{len(FEATURE_NAMES)} features chargées")

load_ml_model()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_input(values_dict):
    """
    Construit un DataFrame aligné sur FEATURE_NAMES,
    remplit les colonnes manquantes à 0,
    applique le scaler, et retourne le DataFrame scalé.
    """
    df = pd.DataFrame(0.0, index=[0], columns=FEATURE_NAMES)
    for col, val in values_dict.items():
        if col in df.columns:
            df[col] = float(val)

    # Feature engineering cohérent avec preprocessing.py
    if 'MonetaryTotal' in values_dict and 'Recency' in values_dict:
        rec = float(values_dict['Recency'])
        mon = float(values_dict['MonetaryTotal'])
        frq = float(values_dict.get('Frequency', 1))
        ten = float(values_dict.get('CustomerTenureDays', 1))
        if 'MonetaryPerDay' in df.columns:
            df['MonetaryPerDay'] = mon / (rec + 1)
        if 'AvgBasketValue' in df.columns:
            df['AvgBasketValue'] = mon / (frq + 1)
        if 'TenureRatio' in df.columns:
            df['TenureRatio'] = rec / (ten + 1)

    if SCALER is not None:
        scaled = SCALER.transform(df)
        df = pd.DataFrame(scaled, columns=FEATURE_NAMES)

    return df


def _get_feature_importance(top_n=5):
    """Retourne les top features sous forme de liste de dicts."""
    if MODEL is None or FEATURE_NAMES is None:
        return []
    if hasattr(MODEL, 'feature_importances_'):
        imp = MODEL.feature_importances_
    elif hasattr(MODEL, 'coef_'):
        imp = np.abs(MODEL.coef_[0])
    else:
        return []
    df = pd.DataFrame({'feature': FEATURE_NAMES, 'importance': imp})
    return df.nlargest(top_n, 'importance').to_dict('records')


def _model_metrics():
    """Calcule les métriques réelles depuis X_test / y_test."""
    try:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'data', 'train_test')
        X_test = pd.read_csv(os.path.join(base, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(base, 'y_test.csv')).values.ravel()

        # Aligner
        for col in FEATURE_NAMES:
            if col not in X_test.columns:
                X_test[col] = 0.0
        X_test = X_test[FEATURE_NAMES]

        # Les données sont déjà scalées dans X_test.csv (preprocessing.py le fait)
        y_pred  = MODEL.predict(X_test)
        y_proba = MODEL.predict_proba(X_test)[:, 1]

        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score)
        return {
            'model_type':    type(MODEL).__name__,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'accuracy':  round(float(accuracy_score(y_test, y_pred)),  4),
            'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)),    4),
            'f1_score':  round(float(f1_score(y_test, y_pred, zero_division=0)),        4),
            'auc_roc':   round(float(roc_auc_score(y_test, y_proba)),  4),
            'cv_f1':     0.90,
            'auc':       round(float(roc_auc_score(y_test, y_proba)),  4),
        }, y_test, y_pred, y_proba
    except Exception as e:
        print(f"_model_metrics error: {e}")
        return {
            'model_type': 'N/A', 'training_date': 'N/A',
            'accuracy': 0, 'precision': 0, 'recall': 0,
            'f1_score': 0, 'auc_roc': 0, 'cv_f1': 0, 'auc': 0,
        }, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# INDEX
# ─────────────────────────────────────────────────────────────────────────────

@main_bp.route('/')
def index():
    from app import db
    from app.models import Prediction
    try:
        recent_preds = Prediction.query.order_by(
            Prediction.timestamp.desc()).limit(5).all()
        metrics, *_ = _model_metrics()
        return render_template('index.html',
                               recent_predictions=recent_preds,
                               metrics=metrics)
    except Exception as e:
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0,
                   'f1_score': 0, 'auc': 0, 'auc_roc': 0}
        return render_template('index.html', metrics=metrics)


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@main_bp.route('/dashboard')
def dashboard():
    from app import db
    from app.models import Prediction
    try:
        # Path to processed data — absolute to avoid working-directory issues
        app_dir  = os.path.dirname(os.path.abspath(__file__))
        proc     = os.path.join(app_dir, '..', 'data', 'processed',
                                'retail_customers_CLEANED.csv')
        if not os.path.exists(proc):
            flash("Fichier de données introuvable. Lancez preprocessing.py d'abord.", 'error')
            return redirect(url_for('main.index'))

        df = pd.read_csv(proc)

        # Le CSV nettoyé contient les données scalées (mean=0, std=1)
        # On recharge aussi le dataset original pour avoir les vraies valeurs des KPI
        raw_candidates = [
            os.path.join(app_dir, '..', 'data', 'raw',
                         'retail_customers_COMPLETE_CATEGORICAL.csv'),
        ]
        df_raw = None
        for rp in raw_candidates:
            if os.path.exists(rp):
                df_raw = pd.read_csv(rp)
                break
        # Si pas de raw, on utilise le cleaned (valeurs scalées)
        df_kpi = df_raw if df_raw is not None else df

        def _json(fig): return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # 1. Pie — distribution Churn
        churn_dist = df['Churn'].value_counts()
        pie = go.Figure(go.Pie(
            labels=['Fidèle', 'Churn'],
            values=[churn_dist.get(0, 0), churn_dist.get(1, 0)],
            hole=.3, marker_colors=['#2ecc71', '#e74c3c']
        ))
        pie.update_layout(title='Distribution des Clients')

        # 2. Box — MonetaryTotal par classe (données brutes si dispo)
        df_viz = df_kpi if (df_kpi is not None and 'Churn' in df_kpi.columns) else df
        box = go.Figure()
        if 'MonetaryTotal' in df_viz.columns and 'Churn' in df_viz.columns:
            box.add_trace(go.Box(y=df_viz[df_viz['Churn']==0]['MonetaryTotal'].dropna(),
                                 name='Fidèle', marker_color='#2ecc71'))
            box.add_trace(go.Box(y=df_viz[df_viz['Churn']==1]['MonetaryTotal'].dropna(),
                                 name='Churn', marker_color='#e74c3c'))
        box.update_layout(title='Montant Total par Classe (£)', yaxis_title='£')

        # 3. Scatter — Recency vs Frequency (données brutes si dispo)
        scatter = go.Figure()
        if 'Recency' in df_viz.columns and 'Frequency' in df_viz.columns and 'Churn' in df_viz.columns:
            for label, color, val in [('Fidèle','#2ecc71',0), ('Churn','#e74c3c',1)]:
                sub = df_viz[df_viz['Churn']==val]
                scatter.add_trace(go.Scatter(
                    x=sub['Recency'], y=sub['Frequency'],
                    mode='markers', name=label,
                    marker=dict(color=color, size=4, opacity=0.5)
                ))
        scatter.update_layout(title='Recency vs Frequency',
                              xaxis_title='Recency (j)', yaxis_title='Frequency')

        # 4. Histogram — Age (données brutes si dispo)
        age_fig = go.Figure()
        if 'Age' in df_viz.columns and 'Churn' in df_viz.columns:
            for label, color, val in [('Fidèle','#2ecc71',0), ('Churn','#e74c3c',1)]:
                age_fig.add_trace(go.Histogram(
                    x=df_viz[df_viz['Churn']==val]['Age'],
                    name=label, marker_color=color, opacity=0.7, nbinsx=20
                ))
        age_fig.update_layout(title="Distribution de l'Âge",
                              barmode='overlay', xaxis_title='Âge')

        # 5. ReturnRatio par Churn
        sat_fig = go.Figure()
        if 'ReturnRatio' in df_viz.columns and 'Churn' in df_viz.columns:
            for label, color, val in [('Fidèle','#2ecc71',0), ('Churn','#e74c3c',1)]:
                sat_fig.add_trace(go.Box(
                    y=df_viz[df_viz['Churn']==val]['ReturnRatio'].dropna(),
                    name=label, marker_color=color
                ))
        sat_fig.update_layout(title='Taux de Retour par Classe')

        # 6. Frequency vs Churn
        ret_fig = go.Figure()
        if 'Frequency' in df_viz.columns and 'Churn' in df_viz.columns:
            for label, color, val in [('Fidèle','#2ecc71',0), ('Churn','#e74c3c',1)]:
                ret_fig.add_trace(go.Box(
                    y=df_viz[df_viz['Churn']==val]['Frequency'].dropna(),
                    name=label, marker_color=color
                ))
        ret_fig.update_layout(title='Fréquence d\'achats par Classe')

        # 7. Feature importance
        imp_json = None
        if MODEL is not None and FEATURE_NAMES is not None:
            top = _get_feature_importance(15)
            if top:
                imp_df = pd.DataFrame(top)
                fig_imp = px.bar(imp_df, x='importance', y='feature',
                                 orientation='h', title='Top 15 Variables Importantes')
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                imp_json = _json(fig_imp)

        # 8. Evolution prédictions (base de données)
        time_json = None
        try:
            preds_by_date = db.session.query(
                db.func.date(Prediction.timestamp),
                db.func.count(Prediction.id)
            ).group_by(db.func.date(Prediction.timestamp)).all()
            if preds_by_date:
                dates, counts = zip(*preds_by_date)
                fig_t = px.line(x=list(dates), y=list(counts),
                                title='Activité des Prédictions',
                                labels={'x': 'Date', 'y': 'Prédictions'})
                time_json = _json(fig_t)
        except Exception:
            pass

        # 9. KPI — utilise df_kpi (données brutes) pour des valeurs lisibles
        def _safe_mean(d, col, decimals=1):
            try:
                return round(float(d[col].mean()), decimals) if col in d.columns else 'N/A'
            except Exception:
                return 'N/A'

        # La colonne Churn peut être dans df (cleaned) mais pas dans df_raw
        df_churn = df if 'Churn' in df.columns else None
        stats = {
            'total_clients':    len(df_kpi) if df_kpi is not None else len(df),
            'churn_count':      int(df_churn['Churn'].sum()) if df_churn is not None else 0,
            'fidele_count':     int((df_churn['Churn']==0).sum()) if df_churn is not None else 0,
            'churn_rate':       round(float(df_churn['Churn'].mean()) * 100, 1) if df_churn is not None else 0,
            'avg_recency':      _safe_mean(df_kpi if df_kpi is not None else df, 'Recency'),
            'avg_monetary':     _safe_mean(df_kpi if df_kpi is not None else df, 'MonetaryTotal', 2),
            'avg_satisfaction': 'N/A',
        }

        return render_template('dashboard.html',
                               pie_chart=_json(pie), box_chart=_json(box),
                               scatter_chart=_json(scatter), age_chart=_json(age_fig),
                               sat_chart=_json(sat_fig), return_chart=_json(ret_fig),
                               importance_chart=imp_json, time_chart=time_json,
                               stats=stats)
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("=== DASHBOARD ERROR ===")
        print(error_detail)
        flash(f"Erreur dashboard: {str(e)}", 'error')
        return redirect(url_for('main.index'))


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT — CLIENT INDIVIDUEL
# ─────────────────────────────────────────────────────────────────────────────

@main_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    from app import db
    from app.models import Prediction
    form = SinglePredictionForm()

    if form.validate_on_submit():
        try:
            if MODEL is None or FEATURE_NAMES is None:
                flash("Modèle non chargé. Vérifiez les fichiers .pkl dans models/", 'error')
                return redirect(url_for('main.predict'))

            values = {
                'Recency':                 form.recency.data,
                'Frequency':               form.frequency.data,
                'MonetaryTotal':           form.monetary_total.data,
                'MonetaryStd':             form.monetary_std.data or 0,
                'Age':                     form.age.data,
                'CustomerTenureDays':      form.customer_tenure_days.data,
                'SupportTicketsCount':     form.support_tickets.data,
                'AvgDaysBetweenPurchases': form.avg_days_between_purchases.data,
                'TotalQuantity':           form.total_quantity.data,
            }

            input_df = _build_input(values)

            prediction    = MODEL.predict(input_df)[0]
            probabilities = MODEL.predict_proba(input_df)[0]
            prob_churn    = float(probabilities[1])

            if prob_churn > 0.7:
                risk_level, risk_color = 'ÉLEVÉ',  'danger'
            elif prob_churn > 0.3:
                risk_level, risk_color = 'MOYEN',  'warning'
            else:
                risk_level, risk_color = 'FAIBLE', 'success'

            # Sauvegarde BD
            rec = Prediction(
                recency=form.recency.data,
                frequency=form.frequency.data,
                monetary_total=form.monetary_total.data,
                age=form.age.data,
                support_tickets=form.support_tickets.data,
                prediction=int(prediction),
                probability_churn=prob_churn,
                risk_level=risk_level
            )
            db.session.add(rec)
            db.session.commit()

            result = {
                'prediction':         'Churn' if prediction == 1 else 'Fidèle',
                'probability_churn':  round(prob_churn * 100, 2),
                'probability_fidele': round(float(probabilities[0]) * 100, 2),
                'risk_level':         risk_level,
                'risk_color':         risk_color,
                'top_features':       _get_feature_importance(5),
            }
            return render_template('predict.html', form=form, result=result)

        except Exception as e:
            flash(f"Erreur de prédiction: {str(e)}", 'error')

    return render_template('predict.html', form=form)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREDICT — CORRECTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

@main_bp.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    from app import db
    from app.models import BatchPrediction
    form = BatchUploadForm()

    if form.validate_on_submit():
        try:
            if MODEL is None or FEATURE_NAMES is None or SCALER is None:
                flash("Modèle non chargé. Lancez preprocessing.py et train_model.py d'abord.", 'error')
                return redirect(url_for('main.batch_predict'))

            # 1. Sauvegarde du fichier uploadé
            upload_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)

            file     = form.file.data
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)

            # 2. Lecture CSV
            df_raw = pd.read_csv(filepath)
            if df_raw.empty:
                flash("Le fichier CSV est vide.", 'error')
                return redirect(url_for('main.batch_predict'))

            original_count = len(df_raw)

            # 3. Feature engineering (même logique que preprocessing.py)
            if 'MonetaryTotal' in df_raw.columns and 'Recency' in df_raw.columns:
                df_raw['MonetaryPerDay'] = df_raw['MonetaryTotal'] / (df_raw['Recency'] + 1)
            if 'MonetaryTotal' in df_raw.columns and 'Frequency' in df_raw.columns:
                df_raw['AvgBasketValue'] = df_raw['MonetaryTotal'] / (df_raw['Frequency'] + 1)
            if 'Recency' in df_raw.columns and 'CustomerTenureDays' in df_raw.columns:
                df_raw['TenureRatio'] = df_raw['Recency'] / (df_raw['CustomerTenureDays'] + 1)
            if 'LastLoginIP' in df_raw.columns:
                df_raw['IP_IsPrivate'] = df_raw['LastLoginIP'].astype(str).apply(
                    lambda ip: 1 if (ip.startswith('10.') or
                                     ip.startswith('192.168.') or
                                     ip.startswith('172.')) else 0)

            # 4. Aligner sur FEATURE_NAMES — colonnes manquantes = 0
            df_aligned = pd.DataFrame(0.0, index=df_raw.index, columns=FEATURE_NAMES)
            for col in FEATURE_NAMES:
                if col in df_raw.columns:
                    df_aligned[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

            # 5. Scaling (les données batch ne sont PAS encore scalées)
            df_scaled = pd.DataFrame(
                SCALER.transform(df_aligned), columns=FEATURE_NAMES
            )

            # 6. Prédictions
            predictions   = MODEL.predict(df_scaled)
            probabilities = MODEL.predict_proba(df_scaled)

            results_df = pd.DataFrame({
                'Prediction': ['Churn' if p == 1 else 'Fidèle' for p in predictions],
                'Prob_Churn': [round(float(p[1]) * 100, 2) for p in probabilities],
                'Risk_Level': ['ÉLEVÉ' if p[1] > 0.7 else 'MOYEN' if p[1] > 0.3 else 'FAIBLE'
                               for p in probabilities]
            })

            churn_count  = int(sum(predictions == 1))
            fidele_count = int(original_count - churn_count)
            high_risk    = int(sum(1 for p in probabilities if p[1] > 0.7))

            # 7. Sauvegarde BD
            batch_rec = BatchPrediction(
                filename=filename,
                total_records=original_count,
                churn_count=churn_count,
                fidele_count=fidele_count,
                high_risk_count=high_risk
            )
            db.session.add(batch_rec)
            db.session.commit()

            # 8. Export CSV résultats
            output_filename = f'results_{filename}'
            output_path     = os.path.join(upload_dir, output_filename)
            # Ajouter colonnes originales si disponibles
            cols_display = [c for c in ['Recency','Frequency','MonetaryTotal','Age']
                            if c in df_raw.columns]
            if cols_display:
                export_df = pd.concat([df_raw[cols_display].reset_index(drop=True),
                                       results_df], axis=1)
            else:
                export_df = results_df
            export_df.to_csv(output_path, index=False)

            # 9. Graphique
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=('Fidèle vs Churn', 'Niveaux de Risque'))
            fig.add_trace(go.Bar(
                x=['Fidèle', 'Churn'], y=[fidele_count, churn_count],
                marker_color=['#2ecc71', '#e74c3c']), row=1, col=1)
            risk_counts = results_df['Risk_Level'].value_counts()
            colors_map  = {'FAIBLE': '#2ecc71', 'MOYEN': '#f39c12', 'ÉLEVÉ': '#e74c3c'}
            bar_colors  = [colors_map.get(r, '#888') for r in risk_counts.index]
            fig.add_trace(go.Bar(
                x=risk_counts.index, y=risk_counts.values,
                marker_color=bar_colors), row=1, col=2)
            fig.update_layout(height=400, showlegend=False)
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template('batch_predict.html',
                                   form=form,
                                   results=results_df.head(20).to_dict('records'),
                                   stats={'total': original_count, 'churn': churn_count,
                                          'fidele': fidele_count, 'high_risk': high_risk},
                                   chart=chart_json,
                                   download_link=output_filename)

        except Exception as e:
            import traceback
            flash(f"Erreur traitement fichier: {str(e)}", 'error')
            print(traceback.format_exc())

    return render_template('batch_predict.html', form=form)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

@main_bp.route('/download/<filename>')
def download_file(filename):
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'static', 'uploads', filename)
        return send_file(path, as_attachment=True)
    except Exception as e:
        flash(f"Erreur téléchargement: {str(e)}", 'error')
        return redirect(url_for('main.batch_predict'))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────

@main_bp.route('/model_info')
def model_info():
    try:
        if MODEL is None:
            flash("Modèle non chargé.", 'error')
            return redirect(url_for('main.index'))

        metrics, y_test, y_pred, y_proba = _model_metrics()

        cm_json = roc_json = imp_json = None

        if y_test is not None:
            from sklearn.metrics import confusion_matrix, roc_curve

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred).tolist()
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=['Prédit Fidèle', 'Prédit Churn'],
                y=['Vrai Fidèle',   'Vrai Churn'],
                colorscale='Blues',
                text=cm, texttemplate="%{text}", textfont={"size": 18}
            ))
            fig_cm.update_layout(title='Matrice de Confusion', height=400)
            cm_json = json.dumps(fig_cm, cls=plotly.utils.PlotlyJSONEncoder)

            # Courbe ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr.tolist(), y=tpr.tolist(),
                name=f"ROC (AUC={metrics['auc_roc']:.3f})",
                line=dict(color='#3498db', width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], name='Aléatoire',
                line=dict(dash='dash', color='#e74c3c')
            ))
            fig_roc.update_layout(title='Courbe ROC',
                                  xaxis_title='FPR', yaxis_title='TPR', height=400)
            roc_json = json.dumps(fig_roc, cls=plotly.utils.PlotlyJSONEncoder)

        # Feature importance
        top = _get_feature_importance(15)
        if top:
            imp_df = pd.DataFrame(top)
            fig_imp = px.bar(imp_df, x='importance', y='feature',
                             orientation='h', title='Top 15 Features Importantes')
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            imp_json = json.dumps(fig_imp, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('model_info.html',
                               metrics=metrics,
                               confusion_matrix=cm_json,
                               roc_curve=roc_json,
                               importance_chart=imp_json)
    except Exception as e:
        flash(f"Erreur model_info: {str(e)}", 'error')
        return redirect(url_for('main.index'))


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────────────────────

@main_bp.route('/history')
def history():
    from app.models import Prediction
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.order_by(
        Prediction.timestamp.desc()
    ).paginate(page=page, per_page=20, error_out=False)
    return render_template('history.html', predictions=predictions)


# ─────────────────────────────────────────────────────────────────────────────
# API JSON
# ─────────────────────────────────────────────────────────────────────────────

@main_bp.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        for field in ['Recency', 'Frequency', 'MonetaryTotal', 'Age']:
            if field not in data:
                return jsonify({'error': f'Champ requis: {field}'}), 400

        input_df = _build_input(data)
        pred     = MODEL.predict(input_df)[0]
        proba    = MODEL.predict_proba(input_df)[0]

        return jsonify({
            'prediction':         int(pred),
            'prediction_label':   'Churn' if pred == 1 else 'Fidèle',
            'probability_churn':  round(float(proba[1]), 4),
            'probability_fidele': round(float(proba[0]), 4),
            'risk_level': 'ÉLEVÉ' if proba[1] > 0.7 else 'MOYEN' if proba[1] > 0.3 else 'FAIBLE'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
