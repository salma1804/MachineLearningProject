import joblib

pca = joblib.load('models/pca_model.pkl')
clf = joblib.load('models/best_classifier.pkl')

print("=== CLF features ===")
print("Nombre:", len(clf.feature_names_in_))

print("\n=== PCA a feature_names_in_ ?", hasattr(pca, 'feature_names_in_'))
if hasattr(pca, 'feature_names_in_'):
    print("Nombre PCA features:", len(pca.feature_names_in_))

print("\n=== Les 5 features engineerees dans CLF ? ===")
for f in ['AvgBasketValue','CancelRate','EngagementScore','MonetaryPerDay','TenureRatio']:
    print(f, "->", f in clf.feature_names_in_)