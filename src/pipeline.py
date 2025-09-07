import sys
import os
sys.path.append(os.path.dirname(__file__))

from zenml import pipeline
from steps.data_loader import load_data
from steps.preprocessor import preprocess_data
from steps.trainer import train_model
from steps.evaluator import evaluate_model

@pipeline
def house_price_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

