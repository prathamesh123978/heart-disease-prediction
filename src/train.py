import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

def train_all(X_train, y_train):
    models = get_models()
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"Trained: {name}")
    return trained

def save_best(model, path='models/best_model.pkl'):
    joblib.dump(model, path)
    print(f"Model saved to {path}")
