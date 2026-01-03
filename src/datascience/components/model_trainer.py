import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import mlflow

from src.datascience.exception import CustomException
from src.datascience.logger import logging
from src.datascience.utils import evaluate_models, save_object

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer initiated")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "RandomForest": RandomForestClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(eval_metric='logloss'),
                "CatBoosting": CatBoostClassifier(verbose=False),
                "AdaBoosting": AdaBoostClassifier()
            }

            params = {
                "DecisionTree": {
                    "criterion": ["gini", "entropy"]
                },
                "RandomForest": {
                    "n_estimators": [100, 200, 300]
                },
                "GradientBoosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200, 300],
                    "subsample": [0.8, 0.9]
                },
                "XGBClassifier": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200, 300]
                },
                "CatBoosting": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200]
                },
                "AdaBoosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model=models,
                params=params,
                scoring="accuracy"
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name} | Accuracy: {best_model_score}")
            print(f"Best Model: {best_model_name}")
            print(f"Accuracy: {best_model_score}")

            model_names = list(params.keys())

            actual_model = ""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]
            mlflow.set_registry_uri("https://dagshub.com/kunjalsunny/Data-Science-End-to-End.mlflow")
            tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
            
            #MLFLOW
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)
                best_model_score = accuracy_score(y_test, predicted_qualities)

                mlflow.log_params(best_params)
                mlflow.log_metric("accuracy", best_model_score)
                mlflow.sklearn.log_model(
                    best_model,
                    artifact_path="model",
                    registered_model_name="Titanic_Classifier_v3"  # This will version your model
                )
            
            if tracking_url_type != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
            else:
                mlflow.sklearn.log_model(best_model, "model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
