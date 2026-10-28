import torch
import torch.nn as nn

# NeuralNetwork
class NeuralNetwork(nn.Module):
    """
        Classe com base em nn.Module para estruturação do modelo inteligênte.\n
        Deve ser utilizado na instanciação de um classe model.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(192)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(192, 120)
        self.bn4 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.pool(self.gelu(self.bn1(self.conv1(x))))
        x = self.pool(self.gelu(self.bn2(self.conv2(x))))
        x = self.pool(self.gelu(self.bn3(self.conv3(x))))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(self.gelu(self.bn4(self.fc1(x))))
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)

        return x
