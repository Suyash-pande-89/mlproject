import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

#decorator
@dataclass
class DataingestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataingestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            #Need to generalize for any data source
            df = pd.read_csv('notebooks\data\StudentsPerformance.csv')
            logging.info("Read the data as dataframe")
            
            #Creating train test directories
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            #os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            #os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            #Writing data into given path
            df.to_csv(self.ingestion_config.raw_data_path,index = False, header=True)
            logging.info("Train Test Split initiated")
            train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)
            train_df.to_csv(self.ingestion_config.train_data_path,index = False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path,index = False, header=True)
            logging.info("Ingestion complete.")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

   
if __name__ == "__main__":
    obj= DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    obj1 = DataTransformation()
    train_arr,test_arr,transformer_path = obj1.initiate_data_transformation(train_path,test_path)
    model_trainer = ModelTrainer()
    trained_model_path = model_trainer.initiate_model_trainer(train_arr,test_arr)
    print(trained_model_path)
    #print(train_arr[0,:],test_arr[0,:])



