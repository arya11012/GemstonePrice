import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import r2_score

def featureAdd(df):
    for i in ['x','y','z']:
        df[i]=df[i].replace(0,df[i].median())
    
    for i in ['x','y','z']:
        df[i]=np.log(df[i])


    df['C/A']=(df['carat'])/(df['depth']*df['table'])
    df['DR']=(df['depth']/df['x'])

    return df
            

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]

            #Train model
            model.fit(X_train,y_train)

            #Predict results
            y_test_pred=model.predict(X_test)


            #test scores
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report
    
    except Exception as e:
        logging.info('Exception occured during model training and testing')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Exception occured in load_object function utils')
        raise CustomException(e,sys)