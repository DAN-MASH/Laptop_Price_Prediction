import pandas as pd
import numpy as np
import os
import sqlite3
import sys


from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
#from sklearn.preprocessing import StandardScaler

@dataclass

class DataIngestionConfig:
    #pass in the datapaths and artifacst that will store the data
    raw_data_path:str=os.path.join("artifacts","raw_data.csv")
    train_data_path:str=os.path.join("artifacts","train_data.csv")
    test_data_path:str=os.path.join("artifacts","test_data.csv")

class DataIngestion:
    try:
        def __init__(self):
            self.data_ingestion_config=DataIngestionConfig()

        def initiate_data_ingestion(self):
            logging.info("Initiate data ingestion")
            #step 1: read the data
            df=pd.read_csv("notebook/data/data/laptop_price - dataset.csv")
            #step 2: Save the data to csv
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            logging.info(f"Data read successfully!")
            #step 3 make sure the train_data directory exists
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            #step 4: split the data into train and test
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
            logging.info(f"train_data:{train_set.shape}")
            logging.info(f"train_data:{test_set.shape}")
            #save the train and test data to csv
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False,header=True)
            logging.info("Data ingestion completed")
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

    except Exception as e:
        logging.info("Error during data ingestion:")
        raise CustomException(e, sys)
      

if __name__=="__main__":
    ingestion=DataIngestion()
    train_data,test_data=ingestion.initiate_data_ingestion()
    
    transformation=DataTransformation()
    train_arr,test_arr,_=transformation.initiate_data_transformation(train_data,test_data)
    