from flask import Flask,jsonify,request
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import json,os

app = Flask(__name__)

valid_datasets=['Social_Network_Ads.csv']

@app.route('/decision-tree',methods=['GET'])
def decision_tree():
    if not request.args.get('dataset'):
        return jsonify(valid_endpoint='/decision-tree?dataset=')
    dataset_name = request.args.get('dataset', type=str)
    if dataset_name not in valid_datasets:
        return jsonify(Error='No data set available',
                       available_datasets='/decision-tree/datasets')

    DIR = os.getcwd()
    DIR = os.path.join(DIR, 'datasets')
    DATASET_PATH = os.path.join(DIR, dataset_name)
    dataset = pd.read_csv(DATASET_PATH)

    column_1='Age'
    column_2='EstimatedSalary'
    #getting  the columns

    X=dataset[[column_1,column_2]].values
    Y=dataset.iloc[:,-1].values #getting the last column from datset


    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #fitting ddata to decision tree
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusionMatrix=pd.DataFrame(cm,columns=["Predicted False","Predicted True"],index=["Actual False","Actual True"])
    results = [{'confusionmatrix' : confusionMatrix}]
    confusionMatrix=pd.Series(results).to_json()
    confusionMatrix=json.loads(confusionMatrix)
    return jsonify(
        confusionMatrix['0']    )

@app.route('/decision-tree/datasets')
def decision_tree_datasets():
    return jsonify(Datasets=valid_datasets)


if __name__ == '__main__':
    app.run()



