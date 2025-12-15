from catboost_model import train_catboost
from tft_model import train_tft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Train models and get predictions
cat_model, X_test, y_test, cat_pred_prob = train_catboost()
tft_model, tft_pred = train_tft()

# Align predictions length
min_len = min(len(cat_pred_prob), len(tft_pred))
cat_pred_prob = cat_pred_prob[:min_len]
tft_pred = tft_pred[:min_len]
y_test = y_test[:min_len]

# Hybrid: Weighted average
final_pred_prob = 0.5*cat_pred_prob + 0.5*tft_pred
final_pred = (final_pred_prob > 0.5).astype(int)

print("Hybrid Model Accuracy:", accuracy_score(y_test, final_pred))
print(classification_report(y_test, final_pred))

# OR Meta-Model (Stacking)
meta_X = np.vstack([cat_pred_prob, tft_pred]).T
meta_model = LogisticRegression()
meta_model.fit(meta_X, y_test)
final_meta_pred = meta_model.predict(meta_X)

print("Meta-Model Accuracy:", accuracy_score(y_test, final_meta_pred))
