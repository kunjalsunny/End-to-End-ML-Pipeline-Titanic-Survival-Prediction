from src.datascience.logger import logging
from src.datascience.exception import CustomException
import sys
from src.datascience.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.datascience.components.data_transformation import DataTransformation, DataTransformationConfig
from src.datascience.components.model_trainer import ModelTrainer, ModelTrainerConfig

if __name__ == "__main__":
    logging.info("The execution started")

    try:
        # data_ingestion_config = DataIngestion.Config()
        data_ingestion = DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
        
        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        # model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

        

    except Exception as e:
        logging.info("Custom exception occurred")
        raise CustomException(e, sys)