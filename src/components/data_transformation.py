import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) :
        self.data_transformation_config = DataTransformationConfig()
        self.num_feat = ["writing score","reading score"]
        self.cat_feat = ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
        self.target_feat = "math score"

    def get_data_transformer_object(self):
        logging.info("Entered the data transformation method or component")
        try:
            #df_train = pd.read_csv(train_path)
            #df_test = pd.read_csv(test_path)
            
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
            logging.info("Pipelines created")

            feat_transformation = ColumnTransformer(
                transformers=[
                    ("num_transformer",num_pipeline,self.num_feat),
                    ("cat_transformer",cat_pipeline,self.cat_feat)
                ]
            )
            logging.info("transformation created")
            logging.info("Transformer Pipeline created")
            return(feat_transformation)
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            
            logging.info("Train and test data read")

            x_train = df_train.drop(columns = [self.target_feat], axis = 1)
            y_train = df_train[self.target_feat]
            
            x_test = df_test.drop(columns = self.target_feat, axis = 1)
            y_test  = df_test[self.target_feat]
            
            logging.info("x and y split")

            feat_transformation = self.get_data_transformer_object()

            x_train_transformed = feat_transformation.fit_transform(x_train)
            x_test_transformed = feat_transformation.transform(x_test)
            
            logging.info("Transformation done")

            train_transformed_arr = np.c_[x_train_transformed,np.array(y_train)]
            #logging.info(x_test_transformed.shape,y_test.shape)
            test_transformed_arr = np.c_[x_test_transformed,np.array(y_test)]


            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as file_obj:
                pickle.dump(feat_transformation, file_obj)

            logging.info("Pickle file created")

            return(train_transformed_arr,test_transformed_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)


