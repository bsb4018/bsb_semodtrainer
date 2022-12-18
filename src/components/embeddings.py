from src.components.data_preprocessing import DataPreprocessing
from src.entity.config_entity import ImageFolderConfig, EmbeddingsConfig
from src.utils.database_handler import MongoDBClient
from torch.utils.data import Dataset, DataLoader
from src.components.model import NeuralNet
from typing import List, Dict
from torchvision import transforms
from collections import namedtuple
from PIL import Image
from torch import nn
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import torch
import os
from pathlib import Path
import sys
from exception.custom_exception import TrainModelException


ImageRecord = namedtuple("ImageRecord", ["img", "label", "s3_link"])


class ImageFolder(Dataset):

    def __init__(self, label_map: Dict):
        try:
            self.config = ImageFolderConfig()
            self.config.LABEL_MAP = label_map
            self.transform = self.transformations()
            self.image_records: List[ImageRecord] = []
            self.record = ImageRecord

            file_list = os.listdir(self.config.ROOT_DIR)

            for class_path in file_list:
                path = os.path.join(self.config.ROOT_DIR, f"{class_path}")
                images = os.listdir(path)
                for image in tqdm(images):
                    image_path = Path(f"""{self.config.ROOT_DIR}/{class_path}/{image}""")
                    self.image_records.append(self.record(img=image_path,
                                                        label=self.config.LABEL_MAP[class_path],
                                                        s3_link=self.config.S3_LINK.format(self.config.BUCKET, class_path,
                                                                                            image)))
        except Exception as e:
            raise TrainModelException(e,sys)

    def transformations(self):
        try:
            TRANSFORM_IMG = transforms.Compose(
                [transforms.Resize(self.config.IMAGE_SIZE),
                transforms.CenterCrop(self.config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]
            )
            return TRANSFORM_IMG

        except Exception as e:
            raise TrainModelException(e,sys)


    def __len__(self):
        try:
            return len(self.image_records)
        except Exception as e:
            raise TrainModelException(e,sys)


    def __getitem__(self, idx):
        try:
            record = self.image_records[idx]
            images, targets, links = record.img, record.label, record.s3_link
            images = Image.open(images)

            if len(images.getbands()) < 3:
                images = images.convert('RGB')
            images = np.array(self.transform(images))
            targets = torch.from_numpy(np.array(targets))
            images = torch.from_numpy(images)

            return images, targets, links
        
        except Exception as e:
            raise TrainModelException(e,sys)



class EmbeddingGenerator:
    def __init__(self, model, device):
        try:
            self.config = EmbeddingsConfig()
            self.mongo = MongoDBClient()
            self.model = model
            self.device = device
            self.embedding_model = self.load_model()
            self.embedding_model.eval()
        except Exception as e:
            raise TrainModelException(e,sys)


    def load_model(self):
        try:
            model = self.model.to(self.device)
            model.load_state_dict(torch.load(self.config.MODEL_STORE_PATH, map_location=self.device))
            return nn.Sequential(*list(model.children())[:-1])
        except Exception as e:
            raise TrainModelException(e,sys)


    def run_step(self, batch_size, image, label, s3_link):
        try:
            records = dict()

            images = self.embedding_model(image.to(self.device))
            images = images.detach().cpu().numpy()

            records['images'] = images.tolist()
            records['label'] = label.tolist()
            records['s3_link'] = s3_link

            df = pd.DataFrame(records)
            records = list(json.loads(df.T.to_json()).values())
            self.mongo.insert_bulk_record(records)

            return {"Response": f"Completed Embeddings Generation for {batch_size}."}
        except Exception as e:
            raise TrainModelException(e,sys)


if __name__ == "__main__":
    try:
        dp = DataPreprocessing()
        loaders = dp.run_step()

        data = ImageFolder(label_map=loaders["valid_data_loader"][1].class_to_idx)
        dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)
        embeds = EmbeddingGenerator(model=NeuralNet(), device="cpu")

        for batch, values in tqdm(enumerate(dataloader)):
            img, target, link = values
            print(embeds.run_step(batch, img, target, link))
    except Exception as e:
            raise TrainModelException(e,sys)

