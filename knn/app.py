from flask import Flask,jsonify,request
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import json
import os
app = Flask(__name__)

valid_datasets=['Social_Network_Ads.csv']
@app.route('/knn')
def knn():
    if not request.args.get('dataset'):
        return jsonify(valid_endpoint='/knn?dataset=')
    dataset_name= request.args.get('dataset', type = str)
    if dataset_name not in valid_datasets:
        return jsonify(Error='No data set available',
                       available_datasets='/knn/datasets')
    DIR = os.getcwd()
    DIR = os.path.join(DIR, 'datasets')
    DATASET_PATH = os.path.join(DIR, dataset_name)
    data=pd.read_csv(DATASET_PATH)
    column_1='Age'
    column_2='EstimatedSalary'
    #getting  the columns

    X=data[[column_1,column_2]].values
    Y=data.iloc[:,-1].values
    total_neighbours=5

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

    #feature scaling

    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)


    #fitting classifier to the training set

    classifier=KNeighborsClassifier(n_neighbors=total_neighbours,)
    classifier.fit(X_train,Y_train)
    Y_predict=classifier.predict(X_test)

    #crsating the confusion matrix


    cm=confusion_matrix(Y_test,Y_predict)
    confusionMatrix=pd.DataFrame(cm,columns=["Predicted False","Predicted True"],index=["Actual False","Actual True"])
    results = [{'confusionmatrix' : confusionMatrix}]
    confusionMatrix=pd.Series(results).to_json()
    confusionMatrix=json.loads(confusionMatrix)
    return jsonify(
       confusionMatrix['0']    )


@app.route('/knn/datasets')
def knnDatasets():
    valid_datasets=['Social_Network_Ads.csv']
    return jsonify(Datasets=valid_datasets)

if __name__ == '__main__':
    app.run()

