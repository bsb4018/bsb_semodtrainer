from src.entity.config_entity import ModelConfig
from src.utils.database_handler import MongoDBClient
from torch import nn
import torch
import sys
from exception.custom_exception import TrainModelException

class NeuralNet(nn.Module):
    def __init__(self):
        try:
            super().__init__()
            self.config = ModelConfig()
            self.mongo = MongoDBClient()
            self.base_model = self.get_model()
            self.labels = self.mongo.get_number_of_labels()
            self.conv1 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.bn1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.bn2 = nn.BatchNorm2d(10)
            self.conv3 = nn.Conv2d(16, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.flatten = nn.Flatten()
            self.final = nn.Linear(4 * 8 * 8, self.labels)
        except Exception as e:
            raise TrainModelException(e,sys)

    def get_model(self):
        try:
            torch.hub.set_dir(self.config.STORE_PATH)
            model = torch.hub.load(
                self.config.REPOSITORY,
                self.config.BASEMODEL,
                pretrained=self.config.PRETRAINED
            )
            return nn.Sequential(*list(model.children())[:-2])

        except Exception as e:
            raise TrainModelException(e,sys)

    def forward(self, x):
        try:
            x = self.base_model(x)
            x = self.conv1(x)
            #x = self.bn1(x)
            x = self.conv2(x)
            #x = self.bn2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            x = self.final(x)
            return x

        except Exception as e:
            raise TrainModelException(e,sys)


if __name__ == '__main__':
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = NeuralNet()
        net.to(device)
    
    except Exception as e:
            raise TrainModelException(e,sys)
