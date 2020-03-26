from flask import Flask,jsonify,request
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os

app = Flask(__name__)


valid_datasets=['Mall_Customers.csv']

@app.route('/kmeans',methods=['GET'])
def kmeans():
    if not request.args.get('dataset'):
        return jsonify(valid_endpoint='/kmeans?dataset=')
    dataset_name = request.args.get('dataset', type=str)
    if dataset_name not in valid_datasets:
        return jsonify(Error='No data set available',
                       available_datasets='/kmeans/datasets')

    DIR = os.getcwd()
    DIR = os.path.join(DIR, 'datasets')
    DATASET_PATH = os.path.join(DIR, dataset_name)
    data = pd.read_csv(DATASET_PATH)

    X=data.iloc[:,[3,4]].values
    kmeans=KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,random_state=0)
    y_means=kmeans.fit_predict(X)
    clusters=[item for item in str(y_means)]
    return jsonify(
        clusters=clusters
    )


@app.route('/kmeans/datasets')
def knnDatasets():
    return jsonify(Datasets=valid_datasets)

if __name__ == '__main__':
    app.run()

