from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.data_loader import load_tabular_data

def train_catboost():
    X, y = load_tabular_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=100)
    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("CatBoost Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test, y_pred_prob
