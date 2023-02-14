#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""script"""
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def get_input(local=False):
    if local:
        print("Reading local file 9c820e0e5b3a4264aa5058f24a82386d.csv")

        return "9c820e0e5b3a4264aa5058f24a82386d.csv"

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename


def run_linear_regression(local=False):
    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return

    iris_data = pd.read_csv(filename, header=0, nrows=12)

    X = iris_data.iloc[:, :1]  # we only take the first two features.

    classes = iris_data.iloc[:, -2]  # assume classes are the final column
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    Y = le.transform(classes)

    # Create an instance of Logistic Regression Classifier and fit the data.
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X, Y)

    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    filename = "logistic_regression.pickle" if local else "/data/outputs/result"
    with open(filename, "wb") as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(Z, pickle_file)


if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_linear_regression(local)


# In[ ]:




