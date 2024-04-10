import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.data_ingestion import DataIngestion
import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) :
        self.data_transformation_config = DataTransformationConfig()
        self.num_feat = ["writing score","reading score"]
        self.cat_feat = ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]

    def get_data_transformer_object(self):
        logging.info("Entered the data transformation method or component")
        try:
            #df_train = pd.read_csv(train_path)
            #df_test = pd.read_csv(test_path)
            logging.info("Read train test data")
            #df_data = pd.concat(df_train,df_test, axis = 1)

            #num_feat = [feat for feat in  df_data.columns if feat.dtype != 'object']
            #cat_feat = [feat for feat in  df_data.columns if feat.dtype == 'object'] 
            #num_feat = self.num_feat
            num_pipeline = Pipeline(
                steps= [
                    ("impute", SimpleImputer(strategy="mean")),
                    ("scale", MinMaxScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("impute",SimpleImputer(strategy = "most_frequent")),
                    ("ohe",OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            feat_transformation = ColumnTransformer(
                transformers=[
                    ("num_transformer",num_pipeline,self.num_feat),
                    ("cat_transformer",cat_pipeline,self.cat_feat)
                ]
            )

            logging.info("Transformer Pipeline created")
            return(feat_transformation)
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)


data_transformed = feat_transformation.fit_transform(df_data)
        data_transformed.to_csv(self.ingestion_config.transform_data_path,index = False, header=True)
            return()



