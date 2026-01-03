import sys
import os
from src.datascience.exception import CustomException
from src.datascience.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.datascience.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")

            numerical_columns = ['age','fare','family_size']
            categorical_columns = ['sex','embarked_Q','embarked_S','isAlone']

            num_pipeline = Pipeline([
                ("Imputer",SimpleImputer(strategy="median")),
                ("Scaler",StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ("Imputer", SimpleImputer(strategy="most_frequent")),
                ("OneHotEncoder", OneHotEncoder(drop='first',handle_unknown='ignore')),
                ("Scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data are loaded")


            #Replace 0 values in 'age' column with NaN
            train_df['age'] = train_df['age'].replace(0,np.nan)
            test_df['age'] = test_df['age'].replace(0,np.nan)
            logging.info("Replaced 0 values in 'age' column with NaN")

            # Fill Embarked missing values with mode
            train_df['embarked'] = train_df['embarked'].fillna(train_df['embarked'].mode()[0])
            test_df['embarked'] = test_df['embarked'].fillna(test_df['embarked'].mode()[0])
            logging.info("Filled missing values in 'embarked' column with mode")

            #One hot-encoding
            for df in [train_df, test_df]:
                df['embarked_Q'] = np.where(df['embarked']=='Q',1,0)
                df['embarked_S'] = np.where(df['embarked']=='S',1,0)
            
            # Feature Engineering
            for df in [train_df, test_df]:
                df['family_size'] = df['sibsp'] + df['parch'] + 1
                df['isAlone'] = np.where(df['family_size']>1,0,1)
            
            target_column = 'survived'
            features = ['age','fare','family_size','sex','embarked_Q','embarked_S','isAlone']


            x_train = train_df[features]
            y_train = train_df[target_column]

            x_test = test_df[features]
            y_test = test_df[target_column]

            logging.info("Applying Preprocessing object on training and testing data")

            prepeprocessor_obj = self.get_data_transformer_object()
            logging.info("Preprocessor pipeline created successfully")

            x_train_arr = prepeprocessor_obj.fit_transform(x_train)
            x_test_arr = prepeprocessor_obj.transform(x_test)

            train_arr = np.c_[x_train_arr, np.array(y_train)]
            test_arr = np.c_[x_test_arr, np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=prepeprocessor_obj
            )

            logging.info("Preprocessor object saved successfully")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path


        except Exception as e:
            raise CustomException(e,sys)