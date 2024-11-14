import pandas as pd
import numpy as np
import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass


@dataclass

class DataTransformationConfig:
   preprocessor_obj_file_path:str= os.path.join("artifacts","preprocessing.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config =DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        try:
            #get the numerical and categorical values
            numerical_columns=[

                "Inches",
                "CPU_Frequency (GHz)",
                "RAM (GB)",
                "Weight (kg)",
                "Price (Euro)"
                ]
            categorical_column=[
                "Company",
                "Product",
                "TypeName",
                "ScreenResolution",
                "CPU_Company",
                "CPU_Type",
                "Memory",
                "GPU_Company",
                "GPU_Type",
                "OpSys"]
            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='mean')),
                    ("scaler",StandardScaler())
                    ])

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("Encoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                    ])
            logging.info(f"Categorical variables: {categorical_column}")
            logging.info(f"Numerical variables {numerical_columns}")
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_column)
                ])
            logging.info("Preprocessor initiated")
            return preprocessor
        
        except Exception as e:
            logging.info("Error occured during preprocessing")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            #step 1 read the data
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
        
            #Step 2: call the preprocessor
            preprocessor_obj=self.get_data_transformation_obj()
            #get the features and target colum
            target_column=["Price (Euro)"]
            input_feature_train=train_df.drop(columns=target_column,axis=1)
            target_train=train_df[target_column]

            input_feature_test=test_df.drop(columns=target_column,axis=1)
            target_test=test_df[target_column]
            #step 3: transform the data
            train_transformed=preprocessor_obj.fit_transform(input_feature_train)
            test_transformed=preprocessor_obj.transform(input_feature_test)
            logging.info(f"Data transformation initiated: Train_transformed {train_transformed.shape},test transformed{test_transformed.shape}")

            train_arr=np.c_[
                train_transformed,np.array(target_train)]
            test_arr = np.c_[
                test_transformed,np.array(target_test)]
            save_object(file_path=self.transformation_config.preprocessor_obj_file_path,
            obj=preprocessor_obj)
            logging.info("Data transformation Complete")
            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)