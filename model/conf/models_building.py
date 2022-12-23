import pandas as pd
import logging
import pickle
import os
import sys
os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
#from data.pg_connector import get_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer

def data_split(df, target_var: str, test_size, the_random_state)->pd.DataFrame:
    '''dividing target var from independent, splitting for train and test'''
    logging.info('Filter out target column')
    X = df.drop(target_var, axis=1)
    y = df[target_var]
    logging.info('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = the_random_state)
    logging.info('Data is splitted')
    return X, y, X_train, X_test, y_train, y_test

def data_scaler(data):

    '''stanrdize data'''

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled,
                                 columns=data.columns)
    return data_scaled

def NB_check(models_array, X_train_scaled, y_train, X_test_scaled, y_test):

    '''finding best naive bayes algorithm'''
  
    for this_model in models_array:

        logging.info('Initializing model')
        clf = this_model()
        logging.info('Training model')
        fitted_model = clf.fit(X_train_scaled, y_train)
        logging.info('Model is trained')
        logging.info('Calculating accuracy')
        print(f'Accuracy of your {this_model} model on training set: {fitted_model.score(X_train_scaled, y_train):.2f}')
        print(f'Accuracy of your {this_model} model on test set: {fitted_model.score(X_test_scaled, y_test):.2f}')
        logging.info('Accuracy is calculated')

def nb_model_fit(model, X_train_scaled, y_train, **kwargs):

    '''fit nb with best params'''

    logging.info('Model initialization')
    clf=model()
    logging.info('Fitting the model')
    fitted_model = clf.fit(X_train_scaled, y_train)
    logging.info('Model is fitted')
    return fitted_model


def best_param_nb(X_test_scaled, y_test):

    '''finding best var_smoothing of the model for model improvement'''

    np.logspace(0,-9, num=10)
    cv_method = RepeatedStratifiedKFold(n_splits=5, 
                                        n_repeats=3, 
                                        random_state=999)
    params_NB1 = {'var_smoothing': np.logspace(0,-9, num=100)}
    
    gs_NB1 = GridSearchCV(estimator=NB1, 
                     param_grid=params_NB1, 
                     cv=cv_method,
                     verbose=1, 
                     scoring='accuracy')
    
    Data_transformed = PowerTransformer().fit_transform(X_test_scaled)
    gs_NB1.fit(Data_transformed, y_test)
    print(gs_NB1.best_params_)

def mean_validation_scores(model, X_train, y_train, cv):
    '''mean validation calculation'''
    scores = cross_val_score(model, X_train, y_train, cv = cv, scoring='accuracy')
    print('Cross-validation scores:{}'.format(scores.mean()))

''''SVM.Checking scores with different kernels'''

def check_kernels(model, X_train_scaled, y_train, X_test_scaled, y_test, array_of_kernels):
    '''see scores with dif kernels'''
    for this_kernel in array_of_kernels:
        logging.info('initializing the model with kernel')
        clf = model(kernel = this_kernel)
    
        logging.info('Training model with kernel')
        clf.fit(X_train, y_train)
    
        logging.info('Calculating accuracy')
        print(f'Accuracy of {this_kernel}-kernel SVC classifier on training set: {clf.score(X_train, y_train):.2f}')
        print(f'Accuracy of {this_kernel}-kernel SVC classifier on test set: {clf.score(X_test, y_test):.2f}')
        logging.info('Accuracies are calculated')

''''SVM model initialization'''

def check_gamma_c(model, X_train_scaled, y_train, X_test_scaled, y_test,  array_of_c, array_of_gamma, kernel_type:str):
    
    for this_gamma in array_of_gamma:

        for this_C in array_of_c:
          logging.info('initializing the model')
          clf = model(kernel = kernel_type, gamma=this_gamma, C = this_C)
          logging.info('Training model with kernel')
          clf.fit(X_train, y_train)
          logging.info('Calculating accuracy')
          print(f'SVM with kernel = {kernel_type}, gamma = {this_gamma} & C = {this_C}')
          print(f'Accuracy of this SVM classifier on training set: {clf.score(X_train, y_train):.2f}')
          print(f'Accuracy of this SVM  classifier on test set: {clf.score(X_test, y_test):.2f}\n')
          logging.info('Accuracies are calculated')