import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps

from enum import Enum

from utils.leaf import LeafCrop

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TransformType(Enum):
    TRAIN = "train"
    EVAL = "eval"


class RandomRot90:
    """Rotaciona 0/90/180/270 graus — folha não tem orientação 'certa' na foto."""
    def __call__(self, img):
        return img.rotate(90 * random.randint(0, 3), expand=True)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def Pil_Loader(path :str, max_side :int=1024) -> Image.Image:
    """
        Loader para o ImageFolder. Em JPEG grande (fotos de 3000px) usa o modo draft,
        que decodifica já reduzido — bem mais rápido. Também aplica a rotação do EXIF.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        img.draft("RGB", (max_side, max_side))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def Get_Transform(type: TransformType, size :int=224):
    """
        TRAIN e EVAL começam com LeafCrop: em fotos de folha inteira sobre fundo claro
        a folha é recortada antes do resize (em close-ups não faz nada).
        EVAL usa Resize((size, size)) em vez de Resize(256) + CenterCrop(224): o CenterCrop
        descartava as bordas da imagem e, numa folha alongada, cortava as pontas.
    """
    if type == TransformType.TRAIN:
        return transforms.Compose([
            LeafCrop(),
            RandomRot90(),
            transforms.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(0.5, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.03),
            # Blur em todas as classes: no dataset base a Rust é mais borrada (imagens 128px ampliadas),
            # sem isso o modelo pode aprender "borrado = Rust"
            transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.1, 2.5))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            # Apaga pedaços aleatórios: força o modelo a usar a folha toda, não um único ponto
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    elif type == TransformType.EVAL:
        return transforms.Compose([
            LeafCrop(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


    
class NeuralNetwork(nn.Module):
    def __init__(self, dropout_rate=.3, num_labels=0):
        super().__init__()

        # Construção das hidden layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 1.2)

        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)

        if num_labels <= 0:
            raise ValueError("It's necessary define a number of labels greater than 0 (num_label:int)")
        else:
            self.fc4 = nn.Linear(128, num_labels) # Saida

        # Ativação GeLU
        self.gelu = nn.GELU()
    
    def forward(self, x):
        # Convoluções
        x = self.pool(self.gelu(self.bn1(self.conv1(x))))
        x = self.pool(self.gelu(self.bn2(self.conv2(x))))
        x = self.pool(self.gelu(self.bn3(self.conv3(x))))
        x = self.pool(self.gelu(self.bn4(self.conv4(x))))


        # Pool Global
        x = self.global_pool(x)
        # Flatten
        x = torch.flatten(x, 1)


        # Dropouts
        x = self.dropout1(self.gelu(self.bn_fc1(self.fc1(x))))
        x = self.dropout1(self.gelu(self.bn_fc2(self.fc2(x))))
        x = self.dropout2(self.gelu(self.fc3(x)))
        x = self.fc4(x)

        return x