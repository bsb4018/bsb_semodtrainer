from src.entity.config_entity import AnnoyConfig
from annoy import AnnoyIndex
from typing_extensions import Literal
from tqdm import tqdm
import json
import sys
from exception.custom_exception import TrainModelException

class CustomAnnoy(AnnoyIndex):
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]):
        try:
            super().__init__(f, metric)
            self.label = []
        except Exception as e:
            raise TrainModelException(e,sys)

    def add_item(self, i: int, vector, label: str) -> None:
        try:
            super().add_item(i, vector)
            self.label.append(label)
        except Exception as e:
            raise TrainModelException(e,sys)

    def get_nns_by_vector(self, vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...):
        try:
            indexes = super().get_nns_by_vector(vector, n, search_k, include_distances)
            labels = [self.label[link] for link in indexes]
            return labels
        except Exception as e:
            raise TrainModelException(e,sys)

    def load(self, fn: str, prefault: bool = ...):
        try:
            super().load(fn)
            path = fn.replace(".ann", ".json")
            self.label = json.load(open(path, "r"))
        except Exception as e:
            raise TrainModelException(e,sys)

    def save(self, fn: str, prefault: bool = ...):
        try:
            super().save(fn)
            path = fn.replace(".ann", ".json")
            json.dump(self.label, open(path, "w"))
        except Exception as e:
            raise TrainModelException(e,sys)


class Annoy(object):
    def __init__(self):
        try:
            self.config = AnnoyConfig()
            self.result = self.mongo.get_collection_documents()["Info"]
        except Exception as e:
            raise TrainModelException(e,sys)

    def build_annoy_format(self):
        try:
            Ann = CustomAnnoy(256, 'euclidean')
            print("Creating Ann for predictions : ")
            for i, record in tqdm(enumerate(self.result), total=8677):
                Ann.add_item(i, record["images"], record["s3_link"])

            Ann.build(100)
            Ann.save(self.config.EMBEDDING_STORE_PATH)
            return True
        except Exception as e:
            raise TrainModelException(e,sys)

    def run_step(self):
        try:
            self.build_annoy_format()
        except Exception as e:
            raise TrainModelException(e,sys)

if __name__ == "__main__":
    try:
        ann = Annoy()
        ann.run_step()
    except Exception as e:
            raise TrainModelException(e,sys)
