from from_root import from_root
import os


class DatabaseConfig:
    def __init__(self):
        self.USERNAME: str = os.environ["DATABASE_USERNAME"]
        self.PASSWORD: str = os.environ["DATABASE_PASSWORD"]
        self.URL: str = "mongodb+srv://<username>:<password>@imagesecluster.pemlurn.mongodb.net/?retryWrites=true&w=majority"
        self.DBNAME: str = "ReverseImageSearchEngine"
        self.COLLECTION: str = "Embeddings"

    def get_database_config(self):
        return self.__dict__


class DataIngestionConfig:
    def __init__(self):
        self.PREFIX: str = "images/"
        self.RAW: str = "data/raw"
        self.SPLIT: str = "data/splitted"
        self.BUCKET: str = "image-database-system-01"
        self.SEED: int = 1337
        self.RATIO: tuple = (0.8, 0.1, 0.1)

    def get_data_ingestion_config(self):
        return self.__dict__