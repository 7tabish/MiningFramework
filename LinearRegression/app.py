from flask import Flask,jsonify,request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import json,os

app = Flask(__name__)

valid_datasets=['USA_Housing.csv']

@app.route('/linearRegression',methods=['GET'])
def linear_regression(): #jsut find the accuracy of predicted model is remaingg
    if not request.args.get('dataset'):
        return jsonify(valid_endpoint='/linearRegression?dataset=')
    dataset_name = request.args.get('dataset', type=str)
    if dataset_name not in valid_datasets:
        return jsonify(Error='No data set available',
                       available_datasets='/linearRegression/datasets')

    DIR = os.getcwd()
    DIR = os.path.join(DIR, 'datasets')
    DATASET_PATH = os.path.join(DIR, dataset_name)
    USAhousing = pd.read_csv(DATASET_PATH)

    X_column='Avg. Area Income'
    Y_column='Price'
    try:
        X = USAhousing[X_column].values.reshape(-1,1)
        y = USAhousing[Y_column].values.reshape(-1,1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #40 % goes to test data
        regressor = LinearRegression()
        regressor.fit(X_train,y_train)
        y_predict=regressor.predict(X_test)
        coefficient=["".join(item) for item in regressor.coef_.astype(str)]
        intercept=["".join(item) for item in regressor.intercept_.astype(str)]
        return jsonify(
            coefficient=coefficient[0],
            Intercept=intercept[0]
        )
    except Exception as error:
        return error


@app.route('/linearRegression/dataset')
def knnDatasets():
    return jsonify(Datasets=valid_datasets)
if __name__ == '__main__':
    app.run()
