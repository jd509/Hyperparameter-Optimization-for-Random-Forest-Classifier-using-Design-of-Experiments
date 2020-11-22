"""
///////////////////////////////////////////////////////////////

Author: Jaineel Desai, University of Southern California

//////////////////////////////////////////////////////////////
"""
from glcm import extract_features
import os

#importing numpy and pandas for computation and storage
import numpy as np
import pandas as pd

#importing modules for supervised learning algorithms
from sklearn.ensemble import RandomForestClassifier

#importing module for computing accuracy and splitting dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


image_dataset_directory = 'Datasets'
glcm_feat = extract_features(image_dataset_directory,1, 0)


#function to capture labels of images (defactorizing labels post classification)
def keep_dict(Y_codes, Y_unique):
    dict = {}
    j = 0
    for i in range(len(Y_codes)):
        if Y_codes[i] in dict:
            continue
        else:
            dict[Y_codes[i]] = Y_unique[j]
            j += 1
    return dict

#random forest classifier training    
def rf_trainer(feat, n_trees, max_feat, max_leaf_nodes, max_depth, min_samples_leaf):
    Y = feat.pop('type')
    X = feat
    Y_codes, Y_unique = pd.factorize(Y) #factorizing labels
    factorized_dict = keep_dict(Y_codes, Y_unique)

    # Make training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(X, Y_codes, test_size=0.25, random_state=42)

    # classify using Random Forest
    clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=25, max_features=max_feat,
                                 max_leaf_nodes=max_leaf_nodes, oob_score=True, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    #fitting data using the classifier
    clf.fit(x_train, y_train)
    return clf, x_test, y_test, factorized_dict

def rf_classifier(n_trees, max_feat, max_leaf_nodes, max_depth, min_samples_leaf):
    global glcm_feat
    clf, x_test, y_test, labels_dict = rf_trainer(glcm_feat,n_trees, max_feat, max_leaf_nodes, max_depth, min_samples_leaf) #Collecting the trained classifier, x_test, y_test and labels
    y_predictions = clf.predict(x_test)  #Predicting the x_test labels
    accuracy = accuracy_score(y_test, y_predictions) #accuracy of prediction
    confusion_matrix_data = confusion_matrix(y_test, y_predictions)
    classification_report_data = classification_report(y_test, y_predictions)
    return accuracy, confusion_matrix_data, classification_report_data


number_of_trees = 100
max_features = 'sqrt'
max_number_of_leafnodes = 5
max_depth_of_tree = 9
min_sample_leaf = 1

accuracy, confusion_matrix, classification_report = rf_classifier(number_of_trees, max_features, max_number_of_leafnodes, max_depth_of_tree, min_sample_leaf)

print(accuracy)
print(confusion_matrix)
print(classification_report)