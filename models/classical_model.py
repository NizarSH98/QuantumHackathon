from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, max_error,
    explained_variance_score, median_absolute_error
)

def train_classical(X_train, X_test, y_train, y_test):
    """
    Trains a RandomForestRegressor and returns multiple regression metrics.

    Returns:
    - metrics (dict): Dictionary with R², RMSE, MAE, Max Error, etc.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "R² Score": round(r2_score(y_test, y_pred), 4),
        "Explained Variance": round(explained_variance_score(y_test, y_pred), 4),
        "Mean Absolute Error (MAE)": round(mean_absolute_error(y_test, y_pred), 4),
        "Median Absolute Error": round(median_absolute_error(y_test, y_pred), 4),
        "Root Mean Squared Error (RMSE)": round(mean_squared_error(y_test, y_pred, squared=False), 4),
        "Max Error": round(max_error(y_test, y_pred), 4)
    }

    return metrics
