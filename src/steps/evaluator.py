from zenml import step
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import logging

@step
def evaluate_model(model, X_test, y_test):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Accuracy: {acc}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return acc, f1, cm



import mlflow

@step
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    return acc
