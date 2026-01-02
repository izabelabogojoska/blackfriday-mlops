import mlflow.pyfunc

def load_model():
    return mlflow.pyfunc.load_model(
        "models:/black_friday_abuse_model/Production"
    )
