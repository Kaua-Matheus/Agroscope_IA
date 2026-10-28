# Torch
import torch

from utils.model import NeuralNetwork
from torchvision import models

from typing import NamedTuple, Optional


"""
Podemos usar diretamente a biblioteca logging
"""
class Colors:
    def __init__(self):
        self.RED = "\033[1;31m"
        self.GREEN = "\033[1;32m"
        self.YELLOW = "\033[1;33m"
        self.BLUE = "\033[1;34m"
        self.DEFAULT = "\033[0m"

    def INFO(self, text:str) -> str:
        return f"[{self.GREEN}INFO{self.DEFAULT}]: {text}"

    def ERROR(self, text:str) -> str:
        return f"[{self.RED}ERROR{self.DEFAULT}]: {text}"


class Response(NamedTuple):
    """
        Modelo e checkpoint

        return:
            Model: Modelo no qual o treinamento foi feito.
            Checkpoint: Estrutura de modelo e informações sobre o mesmo.
    """
    Net: models.ConvNeXt
    Checkpoint: Optional[dict] = None


# LoadModel
def loadModel(model_path: str, device: str = "cpu") -> Response:
    """
        Função para carregamento de modelos treinados.
        Habilitada a opção de carregar os metadados.

        params:
            model_path: Caminho do arquivo .pth do modelo

        Caso existam registros no checkpoint, o primeiro retorno da tupla será o modelo.eval() e o segundo será o checkpoint.
        Senão, somente será retornada uma tupla contendo o modelo.eval()
    """

    colors = Colors()

    try:

        NET = models.convnext_tiny(weights=None)
        in_features = NET.classifier[2].in_features

        LOADED_CHECKPOINT = torch.load(
            model_path,
            map_location=device,
            )

        if LOADED_CHECKPOINT["model_state_dict"] != None:
            NET.classifier[2] = torch.nn.Linear(in_features, LOADED_CHECKPOINT["num_classes"])

            NET.load_state_dict(LOADED_CHECKPOINT["model_state_dict"])

            print(colors.INFO("Modelo c/ checkpoint carregado com sucesso."))
            return Response(NET.eval().to(device), LOADED_CHECKPOINT)
        
        else:
            NET.load_state_dict(torch.load(model_path))

            print(colors.INFO("Modelo s/ checkpoint carregado com sucesso."))
            return Response(NET.eval().to(device), None)


    except Exception as exc:
        print(colors.ERROR(exc))