import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df

def split_and_scale(df, target_col='target', test_size=0.2, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'models/scaler.pkl')
    return X_train, X_test, y_train, y_test
