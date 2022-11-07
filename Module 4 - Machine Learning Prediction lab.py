# Module 4 Python 5 of 5 - Complete the Machine Learning Prediction lab
# Hands-on Lab: Interactive Visual Analytics with Folium
# Guillermo Dominguez

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV

# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression

# Support Vector Machine classification algorithm
from sklearn.svm import SVC

# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree # to plot decision tree
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier

# Libraries to plot Decision Regions
from mlxtend.plotting import plot_decision_regions

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Show all columns in the PyCharm preview
pd.set_option('display.width', 800)  # avoid truncated view
pd.set_option('display.max_columns', 200)  # columns shown
pd.set_option('display.max_rows', 999)  # rows shown

# ===== Function definition
def plot_confusion_matrix(y, y_predict, model):
    # This function plots the confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax) # to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['did not land', 'land'])
    ax.yaxis.set_ticklabels(['did not land', 'landed'])
    plt.suptitle(model+' classification model')
    plt.show()
    plt.tight_layout()
    plt.savefig('Confusion Matrix - '+model+".jpg", dpi=200)
    plt.cla()  # clear instance to free memory

# Loading data (complete table of features (X) and target (Y, Class - Launch success-)
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")
print("Original data size:", data.shape)
print(data.head())

# Load X (one-hot encoded data)
X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv')
print('Imported X data (one-hot encoded), before standarization size:', X.shape)

# TASK 1 -- NumPy array from the column Class
# Create a NumPy array from the column Class in data, by applying the method to_numpy() then assign it to the variable Y
# make sure the output is a Pandas series (only one bracket df['name of column']).

Y = data['Class'].to_numpy() # Class is the launch outcome: 1 successful, 0 failure
print("TASK 1: Original Class (launch outcome) Y data, of size: ", Y.shape)

# TASK 2 -- Standarization of the data
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# Standardize the data in X then reassign it to the variable X using the transform provided below.
# We split the data into training and testing data using the function train_test_split
# The training data is divided into validation data, a second set used for training data;
# then the models are trained and hyperparameters are selected using the function GridSearchCV.

print("Before standardizing, X is:")
print(X.head())
list_columns_X = X.columns

# Standardize features by removing the mean (u) and scaling to unit variance (std. dev, s).
# The standard score of a sample x is calculated as:
# z = (x - u) / s
# Centering and scaling happen independently on each feature by computing the relevant statistics
# on the samples in the training set. Mean and standard deviation are then stored to be used
# on later data using transform.
# class: sklearn.preprocessing.StandardScaler, output is a numpy array (matrix)
# methods:
#   fit (Compute the mean and std to be used for later scaling)
#   fit_transform
#   get_params
#   inverse_transform
#   transform (Perform standardization by centering and scaling)
X = preprocessing.StandardScaler().fit(X).transform(X)
print('The standardized transformed X data of size:', X.shape, 'is:')
print(X)

# TASK 3 -- Split the data into training and testing
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Split the data into training and testing data using the function train_test_split.
# The training data is divided into validation data, a second set used for training data;
# then the models are trained and hyperparameters are selected using the function GridSearchCV.

# Use the function train_test_split to split the data X and Y into training and test data.
# Set the parameter test_size to 0.2 and random_state to 2.
# The training data and test data should be assigned to the following labels.
# X_train, X_test, Y_train, Y_test
# test_size represents the proportion of the dataset to include in the test split
# train_size represents the proportion of the dataset to include in the train split. If None, the value is
# automatically set to the complement of the test size.
# random_state (int) Controls the shuffling applied to the data before applying the split.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print("X train data size:", X_train.shape)
print("X test data size:", X_test.shape)
print("Y train data size:", Y_train.shape)
print("Y test data size (test samples):", Y_test.shape)

# ====================== CLASSIFICATION MODELS ===================

# TASK 4 -- LOGISTIC REGRESSION
# Create a logistic regression object, then create a GridSearchCV object logreg_cv with cv = 10.
# Fit the object to find the best parameters from the dictionary parameters.

# INFO HERE: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
# A regression model regularization techniques
# L1 is called Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function
# L2 is called Ridge Regression. Penalty term is addition of “squared magnitude” of coefficient to loss function.
# Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether.
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
# class sklearn.linear_model.LogisticRegressionCV : implements logistic regression using liblinear, newtoncg, sag or lbfgs (Default) optimizer
# lbfgs: limited-memory BFGS (Broyden–Fletcher–Goldfarb–Shanno) is an optimization algorithm that uses a limited amount of computer memory (https://en.wikipedia.org/wiki/Limited-memory_BFGS)

# dictionary parameters
parameters = {'C': [0.01, 0.1, 1],
             'penalty':['l2'], #l2 originally
             'solver':['lbfgs']} #lbfgs originally
lr = LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters, cv = 10) # CV determines the cross-validation splitting strategy, specify the number of folds
logreg_cv.fit(X_train, Y_train)

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# class sklearn.model_selection.GridSearchCV: Exhaustive search over specified parameter values for an estimator.

# We output the GridSearchCV object for logistic regression (lr).
# We display the best parameters using the data attribute best_params_
# and the accuracy on the validation data using the data attribute best_score_.
# data attribute best_params_ (dictionary): Parameter setting that gave the best results on the hold out data.
# data attribute best_score_ (float): Mean cross-validated score of the best_estimator

print("=== Model Logistic Regression (LR): tuned hyperparameters (best parameters):", logreg_cv.best_params_)

# TASK 5
# Calculate the accuracy on the test data using the method score:
print("LR classification accuracy on the validation data:", round(logreg_cv.best_score_, 5))

# Let's look at the confusion matrix:
yhat = logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat, 'Logistic Regression')
# plt.show()

# Examining the confusion matrix, we see that logistic regression can distinguish between the different classes.
# We see that the major problem is false positives.

# TASK 6 -- SUPPORT VECTOR MACHINE
# Create a support vector machine object, then create a GridSearchCV object svm_cv with cv = 10.
# Fit the object to find the best parameters from the dictionary parameters.

parameters2 = {'kernel': ('poly', 'rbf', 'linear', 'sigmoid'),
              'C': np.logspace(-3, 3, 5), # C hyperparameter: regularization strength C
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(svm, parameters2, cv = 10)
svm_cv.fit(X_train, Y_train)
print("=== Model support vector machine (SVM). Tuned hyperparameters (best parameters):", svm_cv.best_params_)

# TASK 7
# Calculate the accuracy on the test data using the method score:
print("SVM classification accuracy on the validation data:", round(svm_cv.best_score_, 5))

# Plot the confusion matrix
yhat = svm_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat, 'Support Vector Machine')

# Plot Decision regions
def Graph_Decision_Regions(Xinput, Yinput, method, plot_var_1, plot_var_2):
    # method should be trained fitted
    # Only 2D slices of the data can be done
    # SpaceX Falcon 9 has 4 variables: Orbit, Launch Site, Landing Pad & Serial number; numbered as feature 0,1,2 & 3

    # Fill with a fixed value all the rest features not plotted
    # value = 1.5 #dummy value to fix parametes not plotted
    # width = 0.75 #dummy value to fix parametes not plotted
    total_variables = 83
    feature_values = {i: 0.75 for i in range(0, total_variables)}
    feature_values.pop(plot_var_1, None) # remove form the Dict the variable to be ploted
    feature_values.pop(plot_var_2, None) # remove form the Dict the variable to be ploted
    feature_width = {i: 0.25 for i in range(0, total_variables)}
    feature_width.pop(plot_var_1, None) # remove form the Dict the variable to be ploted
    feature_width.pop(plot_var_2, None) # remove form the Dict the variable to be ploted
    # print('CHECK that Dictionary without inputs is correct. Inputs removed:', plot_var_1, plot_var_2)
    fig, ax = plt.subplots()
    plot_decision_regions(Xinput,
                          Yinput,
                          clf=method,
                          feature_index=[plot_var_1, plot_var_2], # feature 1 to be plotted against feature 2
                          filler_feature_values=feature_values,
                          filler_feature_ranges=feature_width,
                          legend=2,
                          ax=ax)  # res=0.02 resolution?

    # https://github.com/rasbt/mlxtend/issues/339

    # Adding axes annotations
    plt.xlabel(list_columns_X[plot_var_1], size=14)
    plt.ylabel(list_columns_X[plot_var_2], size=14)
    plt.title(method + ' on SpaceX Falcon 9 Launch success rate')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()

# Plot the Decision Regions, to make them 2D, variables have to be simplified (filler)
# Still not working properly (filler and features pair selection)
# Graph_Decision_Regions(X_train, Y_train, svm_cv, 78, 82)

#TASK 8  - DECISION TREE
# Create a decision tree classifier object then create a GridSearchCV object tree_cv with cv = 10.
# Fit the object to find the best parameters from the dictionary parameters.
# More details on the hyperparameter of the Decision Tree: https://builtin.com/data-science/train-test-split

parameters3 = {'criterion': ['gini', 'entropy', 'log_loss'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1, 10)], # explores/tunes the hyperparameter depth
     # 'max_features': ['auto', 'sqrt'],
     'max_features': ['sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()
# GFridSearch optimizes the hyperparameters
tree_cv = GridSearchCV(tree, parameters3, cv = 10)
# Decision Tree is trained on the train dataset
tree_cv.fit(X_train, Y_train)

print("=== Model Decision Tree. Tuned hyperparameters (best parameters):", tree_cv.best_params_)

# TASK 9
# Calculate the accuracy of tree_cv on the test data using the method score:
tree_accuracy = round(tree_cv.best_score_, 5)
print("Decision Tree classification accuracy on the validation data:", tree_accuracy)

# Plot the Decision tree
def Plot_Tree(X_input, Y_input):
    plt.figure()
    figure(figsize=(11, 8))  # , dpi=50)
    tree_tuned = DecisionTreeClassifier(criterion=tree_cv.best_params_['criterion'],
                                        max_depth=tree_cv.best_params_['max_depth'],
                                        max_features=tree_cv.best_params_['max_features'],
                                        min_samples_leaf=tree_cv.best_params_['min_samples_leaf'],
                                        min_samples_split=tree_cv.best_params_['min_samples_split'],
                                        splitter=tree_cv.best_params_['splitter']).fit(X_input, Y_input)
    # the darker the color, the more pure that node
    plot_tree(tree_tuned, filled=True, node_ids=True, proportion=True, feature_names=list_columns_X, fontsize=7)
    super_title = "Decision Tree (tuned Hyperparameters) on train data. Accuracy " + str(tree_accuracy)
    plt.suptitle(super_title, fontsize=18)
    plt.title(str(tree_cv.best_params_))
    plt.tight_layout()
    plt.savefig('Decision tree trained.jpg', dpi=500)
    plt.show()
# Graph the Decision tree used after training (on train dataset)
Plot_Tree(X_train, Y_train)

# Plot the confusion matrix on the test dataset, after being optimized for the train dataset
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat, 'Decision Tree')

# TASK 10 -- K-NEAREST NEIGHBOR
# Create a k nearest neighbors object then create a GridSearchCV object knn_cv with cv = 10.
# Fit the object to find the best parameters from the dictionary parameters.
#
parameters4 = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()
knn_cv = GridSearchCV(KNN, parameters4, cv = 10)
knn_cv.fit(X_train, Y_train)

print("=== Model K-Nearest Neighbor tuned hyperparameters (best parameters):", knn_cv.best_params_)

# TASK 11
# Calculate the accuracy of tree_cv on the test data using the method score:
print("KNN classification accuracy on the validation data:", round(knn_cv.best_score_, 5))
# We can plot the confusion matrix

input('Press Enter to continue...')

yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat, 'K-Nearest Neighbors')

# TASK 12 -- Comparison of accuracy
# Find the method performs best:
method = ['Log Reg', 'SVM', 'Tree', 'KNN']
methods_accuracy = [logreg_cv.best_score_, svm_cv.best_score_, tree_cv.best_score_, knn_cv.best_score_]

# fig, ax = plt.subplots(figsize=(8, 6)) #Golden ratio for a more nice appealing size 1.618*width
# ax.bar(method, methods_accuracy, color="blue")

plt.bar(method, methods_accuracy, color="blue")
plt.title('Classification accuracy per method', loc='center')
plt.ylabel('Accuracy')
plt.xlabel('Classification method')
plt.tight_layout()
# plt.show()
plt.savefig('Classification accuracy per method.jpg', dpi=200)

print("==== [[[[[ End of Hand-on Module 4 Machine Learning Prediction lab ]]]] =====")
