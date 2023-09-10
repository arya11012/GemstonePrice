from sklearn.impute import SimpleImputer #Handling missing values
from sklearn.preprocessing import StandardScaler #Feature Scaling
from sklearn.preprocessing import OrdinalEncoder #Encoding categorical variable
from src.utils import featureAdd



#Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

import sys 
import os

from dataclasses import dataclass

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")

            # Categorical and numerical columns
            cat_feature=['cut','color','clarity']
            num_feature=['carat','depth','table','x','y','z']
            imp_num_feature=['carat','y','DR','C/A']
            

            cut_rank=['Fair','Good','Very Good','Premium','Ideal']
            color_rank=['D','E','F','G','H','I','J']
            clarity_rank=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Data Transformation Pipeline Initiated")


            # Numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                        ('imputer',SimpleImputer(strategy='median')),
                        ('scaler',StandardScaler())

                ]
            )

            # Categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                        ('imputer',SimpleImputer(strategy='most_frequent')),
                        ('ordinalencoder',OrdinalEncoder(categories=[cut_rank,color_rank,clarity_rank])),
                        ('scaler',StandardScaler())

                ]
            )


            preprocessor=ColumnTransformer([
                    ('num_pipeline',num_pipeline,imp_num_feature),
                    ('cat_pipeline',cat_pipeline,cat_feature)
            ])

            logging.info("Data Transformation completed")

            return preprocessor

        except Exception as e:
            logging.info("Exception occured at data tranformation stage")
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_data_path,test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("Read train and test data completed")
            logging.info(f'Train Dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe head: \n{test_df.head().to_string()}')


            logging.info('Obtaining Preprocessing object')
            preprocessing_obj=self.get_data_transformation_object()

            target_column='price'
            drop_column=[target_column,'id']

            

            
            
            #Dividing into independent and dependent features

            # Train data
            input_feature_train_df=train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df=train_df[target_column]
            input_feature_train_df=featureAdd(input_feature_train_df)
            


            # Test data
            input_feature_test_df=test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df=test_df[target_column]
            input_feature_test_df=featureAdd(input_feature_test_df)

           

            # Data Transformation


            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



            logging.info("Applying preprocessing object on train and test dataset")

            

        except Exception as e:
            raise CustomException(e,sys)


