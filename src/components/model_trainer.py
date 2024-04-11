import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from dataclasses import dataclass
import pickle

@dataclass
class ModelTrainerConfig():
    model_obj_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        logging.info("Model Training Initiated")
        try:
            model ={
                "LinearRegression" : LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

            x_train = train_arr[:,:-1]
            x_test = test_arr[:,:-1]
            y_train = train_arr[:,-1]
            y_test = test_arr[:,-1]
            logging.info("train test data prepared")

            report = {}
            logging.info("training to find best fit model started")
            for i in range(len(list(model))):
                model_func = list(model.values())[i]
                model_func.fit(x_train,y_train)
                y_pred_test = model_func.predict(x_test)
                report[list(model.keys())[i]]=r2_score(y_test,y_pred_test)

            max_score = max(sorted(report.values()))
            max_score_model_name = list(report.keys())[list(report.values()).index(max_score)]
            
            logging.info("Best Model based on r2 score determined")

            os.makedirs(os.path.dirname(self.model_trainer_config.model_obj_file_path), exist_ok=True)
            with open(self.model_trainer_config.model_obj_file_path, "wb") as file_obj:
                pickle.dump(model[max_score_model_name], file_obj)
            
            logging.info("Model File saved")
            logging.info("Best model is "+max_score_model_name+". Has r2 score :"+ str(max_score) )
            return self.model_trainer_config.model_obj_file_path

        except Exception as e:
            raise CustomException(e,sys)


