import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import csv


def importdata():
    crimedata = pd.read_csv(/home/minix/project 4-2/crime-data_crime-data.csv, sep=",")
    print("dataset length:", len(crimedata))
    print("dataset shape:", crimedata.shape)
    print("dataset", crimedata.head())


def split(crimedata):
    X = crimedata.values[:, 1:5]
    Y = crimedata.values[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 100)
    return X, Y, X_train, X_test, y_train, y_test


//train with gini//


def train_gini(X_train, X_test, y_train):
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    return clf_gini


def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("predicted value:")
    print(y_pred)
    return y_pred


def cal_accuracy(y_test, y_pred):
    print("confusion_matrix: ", confusion_matrix(y_test, y_pred))
    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("report: ", classification_report(y_test, y_pred))


def main():
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = split(data)
    clf_gini = train_gini(X_train, X_test, y_train)
    print("result using gini index: ")
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)


if __name_ == "_main_":
    main()
