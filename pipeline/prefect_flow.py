from prefect import flow, task
from train import train_model

@task
def train():
    train_model()

@flow(name="black-friday-mlops-pipeline")
def pipeline():
    train()

if __name__ == "__main__":
    pipeline()
