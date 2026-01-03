from src.datascience.logger import logging
from src.datascience.exception import CustomException
import sys
from src.datascience.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.datascience.components.data_transformation import DataTransformation, DataTransformationConfig
from src.datascience.components.model_trainer import ModelTrainer, ModelTrainerConfig
from flask import Flask, request, jsonify,render_template
from src.datascience.pipelines.prediction_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            sex=request.form.get('sex'),
            age=int(request.form.get('age')),
            sibsp=int(request.form.get('sibsp')),
            parch=int(request.form.get('parch')),
            fare=float(request.form.get('fare')),
            embarked=request.form.get('embarked')
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Get probability of survival (assuming classifier has predict_proba)
        probability = predict_pipeline.predict_proba(pred_df)[0][1] * 100  # Survival probability
        prediction = "Survived" if results[0] == 1 else "Did Not Survive"

        return render_template('home.html', prediction=prediction, probability=round(probability, 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)

    # logging.info("The execution started")

    # try:
    #     # data_ingestion_config = DataIngestion.Config()
    #     data_ingestion = DataIngestion()
    #     train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
        
    #     data_transformation_config = DataTransformationConfig()
    #     data_transformation = DataTransformation()
    #     train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    #     # model_trainer_config = ModelTrainerConfig()
    #     model_trainer = ModelTrainer()
    #     print(model_trainer.initiate_model_trainer(train_arr,test_arr))

        

    # except Exception as e:
    #     logging.info("Custom exception occurred")
    #     raise CustomException(e, sys)