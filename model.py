
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model_type="Linear Regression", task="regression"):
    """
    Train a machine learning model.

    Parameters:
    - model_type: Model name as string
    - task: "regression" or "classification"

    Returns:
    - Trained model
    """
    model = None

    if task == "regression":
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "Lasso":
            model = Lasso(alpha=0.1)
        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported regression model.")
    
    elif task == "classification":
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "SVM":
            model = SVC(kernel='rbf', probability=True, random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported classification model.")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, task="regression"):
    """
    Evaluate the model using appropriate metrics.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if task == "regression":
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }

    elif task == "classification":
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }

    return metrics

def cross_validate_model(model, X, y, task="regression", cv=5):
    """
    Perform cross-validation and return average score.
    """
    if task == "regression":
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    elif task == "classification":
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

def save_model(model, file_name='climate_model.pkl'):
    """
    Save the trained model to disk.
    """
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name='climate_model.pkl'):
    """
    Load a previously saved models from disk.
    """
    try:
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"[Warning] Model file '{file_name}' not found.")
        return None
