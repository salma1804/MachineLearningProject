from datetime import datetime
from app import db


class Prediction(db.Model):
    """Historique des prédictions individuelles."""
    __tablename__ = 'prediction'

    id                = db.Column(db.Integer, primary_key=True)
    timestamp         = db.Column(db.DateTime, default=datetime.utcnow)

    # Features saisies par l'utilisateur (celles conservées après nettoyage)
    recency           = db.Column(db.Float)
    frequency         = db.Column(db.Float)
    monetary_total    = db.Column(db.Float)
    age               = db.Column(db.Float)
    support_tickets   = db.Column(db.Float)   # SupportTicketsCount

    # Résultat
    prediction        = db.Column(db.Integer)   # 0 = Fidèle, 1 = Churn
    probability_churn = db.Column(db.Float)
    risk_level        = db.Column(db.String(20))

    def to_dict(self):
        return {
            'id':                self.id,
            'timestamp':         self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'recency':           self.recency,
            'frequency':         self.frequency,
            'monetary_total':    self.monetary_total,
            'age':               self.age,
            'prediction':        'Churn' if self.prediction == 1 else 'Fidèle',
            'probability_churn': round(self.probability_churn * 100, 2),
            'risk_level':        self.risk_level,
        }


class BatchPrediction(db.Model):
    """Historique des prédictions par lot (CSV)."""
    __tablename__ = 'batch_prediction'

    id              = db.Column(db.Integer, primary_key=True)
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow)
    filename        = db.Column(db.String(255))
    total_records   = db.Column(db.Integer)
    churn_count     = db.Column(db.Integer)
    fidele_count    = db.Column(db.Integer)
    high_risk_count = db.Column(db.Integer)

    def to_dict(self):
        return {
            'id':              self.id,
            'timestamp':       self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'filename':        self.filename,
            'total_records':   self.total_records,
            'churn_count':     self.churn_count,
            'fidele_count':    self.fidele_count,
            'high_risk_count': self.high_risk_count,
            'churn_rate':      round(self.churn_count / self.total_records * 100, 1)
                               if self.total_records else 0,
        }
