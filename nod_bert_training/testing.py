from datasets import Dataset, load_dataset

DEFAULT_MODEL_NAME = "bert-base-cased"


def create_training_dataset() -> Dataset:
    return load_dataset(path="BI55/MedText")
