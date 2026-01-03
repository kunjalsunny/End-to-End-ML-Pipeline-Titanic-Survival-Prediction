import sys
import os
import pickle
import pandas as pd
from src.datascience.exception import CustomException
from src.datascience.logger import logging
from src.datascience.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)

            preds = model.predict(data_scaled)

            return preds
            
        except Exception as e:
            raise CustomException(e,sys)

    def predict_proba(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            # Prefer predict_proba when available
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(data_scaled)

            # Fallback: use decision_function and convert with softmax
            if hasattr(model, 'decision_function'):
                from scipy.special import softmax
                import numpy as np

                dec = model.decision_function(data_scaled)
                dec = np.atleast_2d(dec)
                return softmax(dec, axis=1)

            # No probability support
            raise CustomException('Model does not support probability estimates', sys)

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, sex: str, age: int, sibsp: int, parch: int, fare: float, embarked: str):
        self.sex = sex
        self.age = age
        self.sibsp = sibsp
        self.parch = parch
        self.fare = fare
        self.embarked = embarked

    def get_data_as_data_frame(self):
        try:
            family_size = self.sibsp + self.parch + 1
            isAlone = 1 if family_size == 1 else 0

            # One-hot encoding internally
            embarked_Q = 1 if self.embarked == "Q" else 0
            embarked_S = 1 if self.embarked == "S" else 0

            data = {
                "age": [self.age],
                "fare": [self.fare],
                "family_size": [family_size],
                "sex": [self.sex],
                "embarked_Q": [embarked_Q],
                "embarked_S": [embarked_S],
                "isAlone": [isAlone]
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
