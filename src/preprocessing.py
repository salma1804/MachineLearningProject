import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Charger les données
df = pd.read_csv('data\\raw\\retail_customers_COMPLETE_CATEGORICAL.csv')

print("=== AVANT NETTOYAGE ===")
print(f"Shape: {df.shape}")
print(f"\nValeurs manquantes par colonne:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nTypes de données:")
print(df.dtypes)


# ============================================
# ÉTAPE 1 : IMPUTATION DES VALEURS MANQUANTES
# ============================================

# 1.1 Age : Imputation par MÉDIANE (1311 valeurs manquantes)
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"Age - Médiane utilisée: {median_age}")

# 1.2 AvgDaysBetweenPurchases : Imputation par MÉDIANE (79 valeurs manquantes)
median_days = df['AvgDaysBetweenPurchases'].median()
df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(median_days)
print(f"AvgDaysBetweenPurchases - Médiane utilisée: {median_days}")

# 1.3 SupportTicketsCount : Remplacer -1 et 999 par NaN, puis MÉDIANE
df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)
print(f"\nSupportTicketsCount - Valeurs uniques avant nettoyage: {sorted(df['SupportTicketsCount'].dropna().unique())[:10]}...")
median_tickets = df['SupportTicketsCount'].median()
df['SupportTicketsCount'] = df['SupportTicketsCount'].fillna(median_tickets)
print(f"SupportTicketsCount - Médiane utilisée: {median_tickets}")

# 1.4 SatisfactionScore : Remplacer -1 et 99 par NaN, puis MODE
df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 99], np.nan)
print(f"\nSatisfactionScore - Valeurs uniques avant nettoyage: {sorted(df['SatisfactionScore'].dropna().unique())}")
mode_satisfaction = df['SatisfactionScore'].mode()[0]
df['SatisfactionScore'] = df['SatisfactionScore'].fillna(mode_satisfaction)
print(f"SatisfactionScore - Mode utilisé: {mode_satisfaction}")

print("\n=== Après imputation ===")
print(f"Valeurs manquantes restantes: {df.isnull().sum().sum()}")


# ============================================
# ÉTAPE 2 : PARSING DES DATES
# ============================================

# Vérifier les formats de date
print("Exemples de RegistrationDate:")
print(df['RegistrationDate'].head(10).tolist())

# Parser les dates avec dayfirst=True (format UK prioritaire)
df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], 
                                         dayfirst=True, 
                                         errors='coerce')

# Extraire les features de date
df['RegYear'] = df['RegistrationDate'].dt.year
df['RegMonth'] = df['RegistrationDate'].dt.month
df['RegDay'] = df['RegistrationDate'].dt.day
df['RegWeekday'] = df['RegistrationDate'].dt.weekday

# Supprimer la colonne RegistrationDate originale
df = df.drop('RegistrationDate', axis=1)

print(f"\nNouvelles features de date créées: RegYear, RegMonth, RegDay, RegWeekday")
print(f"Shape après parsing des dates: {df.shape}")


# ============================================
# ÉTAPE 3 : SUPPRESSION DES FEATURES INUTILES
# ============================================

# 3.1 Supprimer NewsletterSubscribed (toujours "Yes" - pas de variance)
print(f"NewsletterSubscribed - Valeurs uniques: {df['NewsletterSubscribed'].unique()}")
df = df.drop('NewsletterSubscribed', axis=1)
print("NewsletterSubscribed supprimée (valeur constante)")

# 3.2 Supprimer LastLoginIP (trop complexe, pas utile pour ML)
df = df.drop('LastLoginIP', axis=1)
print("LastLoginIP supprimée")

# 3.3 Supprimer CustomerID (identifiant unique, pas utile pour ML)
df = df.drop('CustomerID', axis=1)
print("CustomerID supprimée")

print(f"\nShape après suppression des features inutiles: {df.shape}")


# ============================================
# ÉTAPE 4 : DÉTECTION ET TRAITEMENT DES OUTLIERS
# ============================================

def cap_outliers_iqr(series):
    """Limite les outliers aux bornes IQR"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)

# Colonnes numériques à traiter pour les outliers
numeric_cols_for_outliers = ['MonetaryTotal', 'MonetaryAvg', 'MonetaryStd', 
                              'MonetaryMin', 'MonetaryMax', 'TotalQuantity',
                              'AvgQuantityPerTransaction', 'Age']

print("=== Traitement des outliers (méthode IQR) ===")
for col in numeric_cols_for_outliers:
    if col in df.columns:
        before_min, before_max = df[col].min(), df[col].max()
        df[col] = cap_outliers_iqr(df[col])
        after_min, after_max = df[col].min(), df[col].max()
        print(f"{col}: [{before_min:.2f}, {before_max:.2f}] → [{after_min:.2f}, {after_max:.2f}]")

print("\nOutliers traités avec succès")


# ============================================
# ÉTAPE 5 : GESTION DE LA MULTICOLINÉARITÉ
# ============================================

# Calculer la matrice de corrélation pour les variables numériques
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()

# Trouver les paires fortement corrélées (>0.85)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.85:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print("=== Paires fortement corrélées (|r| > 0.85) ===")
for pair in high_corr_pairs:
    print(f"{pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")

# Supprimer les variables redondantes (conserver celle avec plus de sens métier)
# MonetaryAvg est fortement corrélée avec MonetaryTotal → supprimer MonetaryAvg
if 'MonetaryAvg' in df.columns:
    df = df.drop('MonetaryAvg', axis=1)
    print("\nMonetaryAvg supprimée (redondante avec MonetaryTotal)")

# TotalTransactions et Frequency sont probablement corrélées
if 'TotalTransactions' in df.columns and 'Frequency' in df.columns:
    corr_tf = df['TotalTransactions'].corr(df['Frequency'])
    print(f"\nCorrelation TotalTransactions-Frequency: {corr_tf:.3f}")
    if abs(corr_tf) > 0.85:
        df = df.drop('TotalTransactions', axis=1)
        print("TotalTransactions supprimée (redondante avec Frequency)")

print(f"\nShape après gestion multicolinéarité: {df.shape}")


# Supprimer les variables fortement corrélées (garder celles avec plus de sens métier)

# Frequency ↔ UniqueInvoices (corrélation 1.0) → garder Frequency, supprimer UniqueInvoices
df = df.drop('UniqueInvoices', axis=1)
print("UniqueInvoices supprimée (corrélation 1.0 avec Frequency)")

# UniqueProducts ↔ UniqueDescriptions (corrélation 1.0) → garder UniqueProducts
df = df.drop('UniqueDescriptions', axis=1)
print("UniqueDescriptions supprimée (corrélation 1.0 avec UniqueProducts)")

# MinQuantity ↔ MaxQuantity (corrélation 0.961) → garder MaxQuantity
df = df.drop('MinQuantity', axis=1)
print("MinQuantity supprimée (corrélation 0.961 avec MaxQuantity)")

# AvgProductsPerTransaction ↔ AvgLinesPerInvoice (corrélation 0.963) → garder AvgLinesPerInvoice
df = df.drop('AvgProductsPerTransaction', axis=1)
print("AvgProductsPerTransaction supprimée (corrélation 0.963 avec AvgLinesPerInvoice)")

# NegativeQuantityCount ↔ CancelledTransactions (corrélation 1.0) → garder CancelledTransactions
df = df.drop('NegativeQuantityCount', axis=1)
print("NegativeQuantityCount supprimée (corrélation 1.0 avec CancelledTransactions)")

print(f"\nShape après suppression des variables corrélées: {df.shape}")



# ============================================
# ÉTAPE 6 : ENCODAGE DES VARIABLES CATÉGORIELLES
# ============================================

# Identifier les colonnes catégorielles
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Variables catégorielles à encoder: {categorical_cols}")
print(f"\nNombre de variables catégorielles: {len(categorical_cols)}")

# Vérifier les valeurs uniques pour chaque colonne catégorielle
print("\n=== Valeurs uniques par variable catégorielle ===")
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")


# 6.1 Encodage ORDINAL pour les variables avec ordre naturel

# SpendingCategory: Low < Medium < High < VIP
spending_map = {'Low': 0, 'Medium': 1, 'High': 2, 'VIP': 3}
df['SpendingCategory'] = df['SpendingCategory'].map(spending_map)
print("SpendingCategory encodée (ordinal): Low=0, Medium=1, High=2, VIP=3")

# LoyaltyLevel: Nouveau < Jeune < Établi < Ancien
loyalty_map = {'Nouveau': 0, 'Jeune': 1, 'Établi': 2, 'Ancien': 3}
df['LoyaltyLevel'] = df['LoyaltyLevel'].map(loyalty_map)
print("LoyaltyLevel encodée (ordinal): Nouveau=0, Jeune=1, Établi=2, Ancien=3")

# ChurnRiskCategory: Faible < Moyen < Élevé < Critique
churnrisk_map = {'Faible': 0, 'Moyen': 1, 'Élevé': 2, 'Critique': 3}
df['ChurnRiskCategory'] = df['ChurnRiskCategory'].map(churnrisk_map)
print("ChurnRiskCategory encodée (ordinal): Faible=0, Moyen=1, Élevé=2, Critique=3")

# AgeCategory: Inconnu < 18-24 < 25-34 < 35-44 < 45-54 < 55-64 < 65+
age_map = {'Inconnu': 0, '18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6}
df['AgeCategory'] = df['AgeCategory'].map(age_map)
print("AgeCategory encodée (ordinal): Inconnu=0, 18-24=1, ..., 65+=6")

# BasketSizeCategory: Petit < Moyen < Grand
basket_map = {'Petit': 0, 'Moyen': 1, 'Grand': 2}
df['BasketSizeCategory'] = df['BasketSizeCategory'].map(basket_map)
print("BasketSizeCategory encodée (ordinal): Petit=0, Moyen=1, Grand=2")


# 6.2 One-Hot Encoding pour les variables nominales

# Variables à encoder avec one-hot
nominal_cols = ['RFMSegment', 'CustomerType', 'FavoriteSeason', 'PreferredTimeOfDay', 
                'Region', 'WeekendPreference', 'ProductDiversity', 'Gender', 
                'AccountStatus', 'Country']

print(f"Variables nominales à encoder (one-hot): {len(nominal_cols)}")

# Appliquer one-hot encoding
for col in nominal_cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(col, axis=1)
    print(f"{col}: {len(dummies.columns)} colonnes créées")

print(f"\nShape après encodage: {df.shape}")


# ============================================
# ÉTAPE 7 : SÉPARATION TRAIN/TEST
# ============================================

# Séparer X (features) et y (target)
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"\nDistribution de la target:")
print(y.value_counts(normalize=True))

# Split train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n=== Train/Test Split ===")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
    

# ============================================
# ÉTAPE 8 : SCALING DES FEATURES NUMÉRIQUES
# ============================================

# Identifier les colonnes numériques (exclure la target et les one-hot)
numeric_features = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryStd', 
                    'MonetaryMin', 'MonetaryMax', 'TotalQuantity', 
                    'AvgQuantityPerTransaction', 'MaxQuantity', 'CustomerTenureDays',
                    'FirstPurchaseDaysAgo', 'PreferredDayOfWeek', 'PreferredHour',
                    'PreferredMonth', 'WeekendPurchaseRatio', 'AvgDaysBetweenPurchases',
                    'UniqueProducts', 'UniqueCountries', 'ZeroPriceCount',
                    'CancelledTransactions', 'ReturnRatio', 'TotalTransactions',
                    'AvgLinesPerInvoice', 'Age', 'SupportTicketsCount', 
                    'SatisfactionScore', 'RegYear', 'RegMonth', 'RegDay', 'RegWeekday',
                    'SpendingCategory', 'LoyaltyLevel', 'ChurnRiskCategory', 
                    'AgeCategory', 'BasketSizeCategory']

# Filtrer pour ne garder que les colonnes qui existent encore
numeric_features = [col for col in numeric_features if col in X_train.columns]

print(f"Features numériques à scaler: {len(numeric_features)}")

# Appliquer StandardScaler (fit sur train, transform sur train et test)
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

print("Scaling appliqué avec StandardScaler")
print(f"\nExemple de valeurs scalées (X_train premier échantillon):")
print(X_train.iloc[0][numeric_features[:5]])
    

# ============================================
# ÉTAPE 9 : SAUVEGARDE DES FICHIERS NETTOYÉS
# ============================================

import os

# Créer le dossier de sortie
output_dir = 'C:\\Users\\User\\Desktop\\MlProject\\MachineLearningProject\\data\\processed'
output_dir1 = 'C:\\Users\\User\\Desktop\\MlProject\\MachineLearningProject\\data\\train_test'

# Sauvegarder les datasets
X_train.to_csv(f'{output_dir1}/X_train.csv', index=False)
X_test.to_csv(f'{output_dir1}/X_test.csv', index=False)
y_train.to_csv(f'{output_dir1}/y_train.csv', index=False)
y_test.to_csv(f'{output_dir1}/y_test.csv', index=False)

# Sauvegarder aussi le dataset complet nettoyé
df_cleaned = pd.concat([X_train, X_test])
df_cleaned['Churn'] = pd.concat([y_train, y_test])
df_cleaned.to_csv(f'{output_dir}/retail_customers_CLEANED.csv', index=False)
print(f"✓ {output_dir}/retail_customers_CLEANED.csv ({df_cleaned.shape[0]} lignes, {df_cleaned.shape[1]} colonnes)")