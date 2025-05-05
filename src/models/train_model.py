from joblib import dump

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

def train_and_save_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 52)

    dt_params = {
        'max_depth' : [3, 7, 12, 16, 20],
        'min_samples_split' : [2, 7, 12, 16, 20],
        'min_samples_leaf' : [1, 2, 5, 7, 10],
        'max_features' : ['sqrt', 'log2'],
        'max_leaf_nodes' : [10, 25, 50, 75, 100]
    }
    dt_grid = GridSearchCV(DecisionTreeRegressor(), dt_params, cv=5)
    dt_grid.fit(x_train, y_train)
    dt = dt_grid.best_estimator_

    param_grid_gb = {
        'n_estimators': [100, 200],            
        'learning_rate': [0.05, 0.1, 0.2],     
        'max_depth': [3, 5, 7],                
        'min_samples_split': [2, 5, 10],       
        'min_samples_leaf': [1, 2, 4],         
        'subsample': [0.8, 1.0],               
        'max_features': ['sqrt', 'log2']       
    }
    gb_grid = GridSearchCV(GradientBoostingRegressor(), param_grid_gb, scoring='neg_mean_squared_error', cv=5)
    gb_grid.fit(x_train, y_train)
    gb = gb_grid.best_estimator_

    param_grid_rf = {     
        'max_depth': [10, 12, 15],         
        'min_samples_split': [4, 7, 10],         
        'min_samples_leaf': [1, 2, 4],           
        'max_features': ['sqrt', 'log2', 0.5],   
        'bootstrap': [True, False]               
    }
    rf_grid = GridSearchCV(RandomForestRegressor(), param_grid_rf, scoring='neg_mean_squared_error', cv=5)
    rf_grid.fit(x_train, y_train)
    rf = rf_grid.best_estimator_

    dump(dt, '/Users/timurchiks/Desktop/flight_price_predictor/models/joblib/dt_v2.joblib')
    dump(gb, '/Users/timurchiks/Desktop/flight_price_predictor/models/joblib/gb_v2.joblib')
    dump(rf, '/Users/timurchiks/Desktop/flight_price_predictor/models/joblib/rf_v2.joblib')




