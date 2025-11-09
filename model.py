import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression,Ridge,Lasso,ElasticNet,SGDRegressor,HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline #data leakage
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as  lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle

data = pd.read_csv(r"C:\Users\AR ANSARI\NIT\ML\vs\Regression Project\USA_Housing (model).csv")

# preprocessing
X = data.drop(['Price', 'Address'], axis=1)
Y = data['Price']

#split 
X_train,X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

# Define  model
models = {
    "LinearRegression": LinearRegression(),
    "RobustRegression": HuberRegressor(),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "ElasticNet": ElasticNet(),
    "PolynomialRegression": Pipeline([('poly', PolynomialFeatures(degree=4)), ('linear', LinearRegression())]),
    "SGDRegressor": SGDRegressor(),
    "ANN": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    "RandomForest": RandomForestRegressor(),
    "SVM": SVR(),
    "LGBM": lgb.LGBMRegressor(),
    "XGBoost": xgb.XGBRegressor(),
    "KNN": KNeighborsRegressor(),
}

# train and evaluate model
results = []

for name,model in models.items():
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)

    mae = mean_absolute_error(Y_test,Y_pred)
    mse = mean_squared_error(Y_test,Y_pred)
    r2 =  r2_score(Y_test,Y_pred)

    results.append({
        'Model' : name,
        'MAE' : mae,
        'MSE' :mse,
        'R2' : r2

    })

    with open(f"{name}.pkl",'wb') as f:
        pickle.dump(model,f)

# convert result to Dataframe and save to csv

results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print("Model have been trained and saved as pickle files. evaluation result have been saved to model_evaluation_result.csv.")