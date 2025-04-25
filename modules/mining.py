import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


def classify_rf(df, target_col, test_size=0.2):
    # Ensure target is string (robust for category dtype)
    y = df[target_col].astype(str)
    # One-hot encode categorical features, drop target
    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred)
    }

def regress_linear(df, y_col, x_col, test_size=0.2):
    # Use only the specified columns
    X = df[[x_col]]
    y = df[y_col]
    # Ensure both are numeric
    if not pd.api.types.is_numeric_dtype(X[x_col]):
        raise ValueError(f"X column '{x_col}' must be numeric for regression.")
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError(f"Y column '{y_col}' must be numeric for regression.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return reg.score(X_test, y_test), reg

def cluster_kmeans(df, n_clusters=3):
    # Convert categorical variables to binary (one-hot)
    X = pd.get_dummies(df.select_dtypes(include=['number', 'category']), drop_first=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels
