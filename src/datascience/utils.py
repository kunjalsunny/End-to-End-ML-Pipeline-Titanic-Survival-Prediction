import os
import sys
from src.datascience.exception import CustomException
from src.datascience.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

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