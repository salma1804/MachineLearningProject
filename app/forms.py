from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Optional


class SinglePredictionForm(FlaskForm):
    """
    Formulaire de prédiction individuelle.
    Contient uniquement les features conservées après le nettoyage anti-leakage :
    - Supprimées : SatisfactionScore, SpendingCategory, LoyaltyLevel,
                   AgeCategory, BasketSizeCategory, ChurnRiskCategory,
                   RFMSegment, CustomerType, CancelledTrans, ZeroPriceCount
    """

    recency = FloatField(
        'Récence — jours depuis le dernier achat',
        validators=[DataRequired(), NumberRange(min=0, max=1000)],
        render_kw={"placeholder": "Ex: 45"}
    )
    frequency = FloatField(
        'Fréquence — nombre de commandes',
        validators=[DataRequired(), NumberRange(min=1, max=1000)],
        render_kw={"placeholder": "Ex: 12"}
    )
    monetary_total = FloatField(
        'Montant total dépensé (£)',
        validators=[DataRequired(), NumberRange(min=0)],
        render_kw={"placeholder": "Ex: 1250.50"}
    )
    monetary_std = FloatField(
        'Écart-type des montants (£)',
        validators=[Optional()],
        render_kw={"placeholder": "Ex: 85.00"}
    )
    age = FloatField(
        'Âge du client',
        validators=[DataRequired(), NumberRange(min=18, max=100)],
        render_kw={"placeholder": "Ex: 35"}
    )
    customer_tenure_days = FloatField(
        'Ancienneté (jours)',
        validators=[DataRequired(), NumberRange(min=0)],
        render_kw={"placeholder": "Ex: 365"}
    )
    support_tickets = IntegerField(
        'Nombre de tickets support',
        validators=[DataRequired(), NumberRange(min=0, max=50)],
        render_kw={"placeholder": "Ex: 2"}
    )
    avg_days_between_purchases = FloatField(
        'Moyenne jours entre achats',
        validators=[DataRequired(), NumberRange(min=0)],
        render_kw={"placeholder": "Ex: 30"}
    )
    total_quantity = FloatField(
        'Quantité totale achetée',
        validators=[DataRequired()],
        render_kw={"placeholder": "Ex: 150"}
    )
    return_ratio = FloatField(
        'Taux de retour (0.0 à 1.0)',
        validators=[Optional(), NumberRange(min=0, max=1)],
        render_kw={"placeholder": "Ex: 0.05"}
    )
    unique_products = IntegerField(
        'Nombre de produits différents achetés',
        validators=[Optional(), NumberRange(min=0)],
        render_kw={"placeholder": "Ex: 15"}
    )

    submit = SubmitField('Prédire le Churn')


class BatchUploadForm(FlaskForm):
    """
    Formulaire d'upload CSV pour prédiction par lot.
    Le CSV doit contenir au minimum : Recency, Frequency, MonetaryTotal, Age.
    Les colonnes manquantes seront automatiquement remplies à 0.
    """
    file = FileField(
        'Fichier CSV',
        validators=[
            DataRequired(),
            FileAllowed(['csv'], 'Fichiers CSV uniquement !')
        ]
    )
    submit = SubmitField('Analyser le fichier')
