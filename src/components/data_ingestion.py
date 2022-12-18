from src.entity.config_entity import DataIngestionConfig
from src.utils.storage_handler import S3Connector
from from_root import from_root
import splitfolders
import os
import sys
from exception.custom_exception import TrainModelException

class DataIngestion:
    def __init__(self):
        try:
            self.config = DataIngestionConfig()
        except Exception as e:
            raise TrainModelException(e,sys)

    def download_dir(self):
        """
        params:
        - prefix: pattern to match in s3
        - local: local path to folder in which to place files
        - bucket: s3 bucket with target contents
        - client: initialized s3 client object

        """
        try:
            print("\n====================== Fetching Data ==============================\n")
            data_path = os.path.join(from_root(), self.config.RAW, self.config.PREFIX)
            os.system(f"aws s3 sync s3://image-database-system-01/images/ {data_path} --no-progress")
            print("\n====================== Fetching Completed ==========================\n")

        except Exception as e:
            raise TrainModelException(e,sys)

    def split_data(self):
        """
        This Method is Responsible for splitting.
        :return:
        """
        try:
            splitfolders.ratio(
                input=os.path.join(self.config.RAW, self.config.PREFIX),
                output=self.config.SPLIT,
                seed=self.config.SEED,
                ratio=self.config.RATIO,
                group_prefix=None, move=False
            )
        except Exception as e:
            raise TrainModelException(e,sys)

    def run_step(self):
        try:
            self.download_dir()
            self.split_data()
            return {"Response": "Completed Data Ingestion"}
        except Exception as e:
            raise TrainModelException(e,sys)

if __name__ == "__main__":
    try:
        paths = ["data", r"data\raw", r"data\splitted", r"data\embeddings",
                 "model", r"model\benchmark", r"model\finetuned"]

        for folder in paths:
            path = os.path.join(from_root(), folder)
            print(path)
            if not os.path.exists(path):
                os.mkdir(folder)

        dc = DataIngestion()
        print(dc.run_step())
    except Exception as e:
        raise TrainModelException(e,sys)
