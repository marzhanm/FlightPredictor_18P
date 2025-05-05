from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json

def evaluate_model(x, y):
    try:
        dt = load('/Users/timurchiks/Desktop/flight_price_predictor/models/joblib/dt_v2.joblib')
        gb = load('/Users/timurchiks/Desktop/flight_price_predictor/models/joblib/gb_v2.joblib')
        rf = load('/Users/timurchiks/Desktop/flight_price_predictor/models/joblib/rf_v2.joblib')
    except Exception:
        print("не удалось загрузить модель")
    
    x_tree_train, x_tree_test, y_tree_train, y_tree_test = train_test_split(x, y, test_size=0.2, random_state=52)

    dt_pred = dt.predict(x_tree_test)
    dt_metrics = {
        'MAE' : mean_absolute_error(y_tree_test, dt_pred),
        'RMSE' : np.sqrt(mean_squared_error(y_tree_test, dt_pred)),
        'R2' : r2_score(y_tree_test, dt_pred)
    }
    print('-----------dt-----------')
    print("Mean Squared Error (MSE):", mean_squared_error(y_tree_test, dt_pred))
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_tree_test, dt_pred))
    print("test R² Score:", r2_score(y_tree_test, dt_pred))

    gb_pred = gb.predict(x_tree_test)
    gb_metrics = {
        'MAE' : mean_absolute_error(y_tree_test, gb_pred),
        'RMSE' : np.sqrt(mean_squared_error(y_tree_test, gb_pred)),
        'R2' : r2_score(y_tree_test, gb_pred)
    }
    print('-----------gb-----------')
    print("Mean Squared Error (MSE):", mean_squared_error(y_tree_test, gb_pred))
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_tree_test, gb_pred))
    print("test R² Score:", r2_score(y_tree_test, gb_pred))

    rf_pred = rf.predict(x_tree_test)
    rf_metrics = {
        'MAE' : mean_absolute_error(y_tree_test, rf_pred),
        'RMSE' : np.sqrt(mean_squared_error(y_tree_test, rf_pred)),
        'R2' : r2_score(y_tree_test, rf_pred)
    }
    print('-----------rf-----------')
    print("Mean Squared Error (MSE):", mean_squared_error(y_tree_test, rf_pred))
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_tree_test, rf_pred))
    print("test R² Score:", r2_score(y_tree_test, rf_pred))


    with open("/Users/timurchiks/Desktop/flight_price_predictor/models/metrics/dt_metrics_v2.json", "w") as f:
        json.dump(dt_metrics, f)
    with open("/Users/timurchiks/Desktop/flight_price_predictor/models/metrics/gb_metrics_v2.json", "w") as f:
        json.dump(gb_metrics, f)
    with open("/Users/timurchiks/Desktop/flight_price_predictor/models/metrics/rf_metrics_v2.json", "w") as f:
        json.dump(rf_metrics, f)
    

    