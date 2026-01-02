import os
import sys
from src.datascience.exception import CustomException
from src.datascience.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.datascience.utils import evaluate_models, save_object

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Model Trainer initiated")
            
            logging.info("Splitting training and test input data")
            X_train=train_array[:,:-1]
            y_train=train_array[:,-1]
            X_test=test_array[:,:-1]
            y_test=test_array[:,-1]


            models = {
                'RandomForest': RandomForestRegressor(),
                'DecisionTree': DecisionTreeRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoosting': CatBoostRegressor(verbose=False),
                'AdaBoosting': AdaBoostRegressor()
            }

            params = {
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "CatBoosting":{
                    'depth':[6,8,10],
                    'learning_rate':[.1,.01,0.05,.001],
                    'iterations':[30,50,100]
                },
                "AdaBoosting":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,params=params,model=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print("This is the best model:")
            print(best_model_name)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            r2_square = r2_score(y_test,best_model.predict(X_test))
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)