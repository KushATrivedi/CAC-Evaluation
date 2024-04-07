#imports 
from sklearn import linear_model, tree, ensemble
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# list of models
models = {
    #"linear_regression" : linear_model.LinearRegression(),
    "ridge":linear_model.Ridge(),
    "lasso":linear_model.Lasso(),
    "decision_tree" : tree.DecisionTreeRegressor(),
    "random_forest" : ensemble.RandomForestRegressor(),
    "xgboost" : XGBRegressor(),
    "lgbm": LGBMRegressor(),
    "catboost": CatBoostRegressor(),
}