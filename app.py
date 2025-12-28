from src.datascience.logger import logging
from src.datascience.exception import CustomException
import sys
from src.datascience.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    logging.info("The execution started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
        
    except Exception as e:
        logging.info("Customer exception occurred")
        raise CustomException(e, sys)