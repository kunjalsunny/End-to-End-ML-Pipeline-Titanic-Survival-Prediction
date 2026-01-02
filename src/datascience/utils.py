import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.datascience.exception import CustomException
from src.datascience.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db_name = os.getenv('db_name')


def read_sql_data():
    logging.info('Reading SQL')
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db_name
        )
        logging.info("Connection established successfully",mydb)
        df=pd.read_sql_query("SELECT * FROM titanic",mydb)
        print(df.head(5))
        logging.info("SQL data read as dataframe")
        return df
        
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, model, params):
    try:
        report = {}

        for model_name, model_obj in model.items():
            para = params.get(model_name)

            if para:
                gs = GridSearchCV(model_obj, para, cv=3)
                gs.fit(X_train, y_train)
                model_obj.set_params(**gs.best_params_)

            # Fit the (possibly tuned) model
            model_obj.fit(X_train, y_train)

            y_train_pred = model_obj.predict(X_train)
            y_test_pred = model_obj.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)