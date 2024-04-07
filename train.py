#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

import joblib
import os
import argparse
import config
import model_dispatcher


def save_charts(chart, chart_name):

    if not os.path.exists(config.CHARTS_OUTPUT):
        os.makedirs(config.CHARTS_OUTPUT)
    chart.savefig(os.path.join(config.CHARTS_OUTPUT, f"{chart_name}.png"))


def hist_cost(df):
    plt.figure(figsize=(8, 6))
    sns.histplot(data = df['cost'], kde =True, color='skyblue')
    plt.title('Histogram of Cost')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.grid(True)

    save_charts(plt, "histogram_of_cost")


def box_plot(df):
    df.plot(kind="box",subplots=True,  layout = (5,8), figsize = (30,30))
    plt.subplots_adjust(wspace = 0.5)

    save_charts(plt, "boxplots")

def  plot_correlation(df, num_vars):
    plt.figure(figsize=(15,10))
    plt.title("Correlation matrix")
    mask = np.triu(np.ones_like(df[num_vars].corr(), dtype=bool))
    sns.heatmap(df[num_vars].corr(), vmax=1.0, vmin = -1.0, mask=mask, annot=True, fmt=".2f", cmap='coolwarm')

def evaluate_regression_models(model, X_train, y_train, X_val, y_val, X_test, y_test):
    
    #model fitting
    reg = model_dispatcher.models[model]
    reg.fit(X_train, y_train)
    
    y_val_pred = reg.predict(X_val)
    y_test_pred = reg.predict(X_test)
    
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    rmse_val = sqrt(mse_val)
    r2_val = r2_score(y_val, y_val_pred)
    
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    
    scores_df = pd.DataFrame({
        'Model': [model],
        'MAE Validation': [mae_val],
        'MSE Validation': [mse_val],
        'RMSE Validation': [rmse_val],
        'R2 Validation': [r2_val],
        'MAE Test': [mae_test],
        'MSE Test': [mse_test],
        'RMSE Test': [rmse_test],
        'R2 Test': [r2_test]
    })

    # Save the model
    if not os.path.exists(config.MODEL_OUTPUT):
        os.makedirs(config.MODEL_OUTPUT)
    joblib.dump(reg, os.path.join(config.MODEL_OUTPUT, f"{model}.bin"))
    
    return scores_df

def run(model):
    #reading input file
    df = pd.read_csv(config.INPUT_FILE)

    # #preprocessing
    # cat_vars = []
    # num_vars = []
    # for i in df.columns:
    #     if(df[i].dtype == "object"):
    #         cat_vars.append(i)
    #     else:
    #         num_vars.append(i)
    
    #plotting histogram of cost
    hist_cost(df)
    print("Histogram of Cost is ploted and saved")

    #plotting boxplots of continuos variables
    box_plot(df)
    print("Boxplots are ploted and saved")

    #plotting correlation matrix
    num_vars = df.select_dtypes(include=np.number).columns.tolist()
    plot_correlation(df, num_vars)
    print("Correlation Matrix is Ploted and Saved")

    #dropping columns
    columns=["salad_bar", "gross_weight", "avg_cars_at home(approx).1", "store_sales(in millions)", "store_cost(in millions)", "meat_sqft","frozen_sqft", "store_sqft", "grocery_sqft","prepared_food"]
    df2 = df.drop(columns=columns)
    
    #boxplots after dropping columns
    box_plot(df)
    print("Boxplots are dropping columns areploted and saved")

    #saving the new num and cat vars of df2
    cat_vars2 = []
    num_vars2 = []
    for i in df2.columns:
        if(df2[i].dtype == "object"):
            cat_vars2.append(i)
        else:
            num_vars2.append(i)

    #label encoding
    label_encoder = LabelEncoder()

    for column in df2.columns:
        if df2[column].dtype == 'object':  
            df2[column] = label_encoder.fit_transform(df2[column])
    
    #train_test_split
    features = df2.columns.drop("cost")
    target = df2["cost"]

    x_train, x_val_test, y_train, y_val_test = train_test_split(df2[features], target, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5)

    print("Shape of Training X :",x_train.shape)
    print("Shape of Validation X :",x_val.shape)
    print("Shape of Training y :",y_train.shape)
    print("Shape of Validation y :",y_val.shape)
    print("Shape of Testing X :",x_test.shape)
    print("Shape of Testing y :",y_test.shape)

    #models creation:
    print("Running model: ",model)
    scores = evaluate_regression_models(model, x_train, y_train, x_val, y_val, x_test, y_test)
    print(scores)
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #arg1
    parser.add_argument(
        "--model", type = str
    )

    args = parser.parse_args()

    #work
    run(args.model)
