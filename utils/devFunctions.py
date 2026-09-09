# Numpy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# OpenCV
import cv2
import os
from pathlib import Path


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


# Mostra as imagens
def Imshow(img, title :str=""):
    img = img / 2 + 0.5 # Retirando o normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def CropImage(input_path :str, output_path :str=""):
    """
        Corta as imagens de dentro pasta (input) para adquirir somente o objeto (folha) central
    """
    print(Colors().INFO("Iniciando Processamento..."))

    # Lista de sufixos dos arquivos
    suffix = [".jpeg", ".jpg", ".png", "webp"]

    if os.path.isdir(input_path):

        print(f"Pasta atual: {Colors().BLUE}{input_path}{Colors().DEFAULT}")
        src = Path(input_path)
        files_list = list(src.iterdir())

        for (index, file) in enumerate(files_list):
            if index % 1000 == 0 and index != 0:
                print("Aoooo")
                print(f"{(index / len(files_list) * 100):.2f}% - {index} / {len(files_list)}")

            if file.suffix in suffix:

                try:
                    # Manipulação de Imagem
                    img = cv2.imread((input_path + "/" if input_path[-1] != "/" else input_path) + file.name)
                    blue = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    blur = cv2.GaussianBlur(blue, (5, 5), 0)

                    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

                    lower = (25, 40, 40)
                    upper = (85, 255, 255)

                    mask = cv2.inRange(hsv, lower, upper)

                    # Limpeza de Máscara
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    # Contornos
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    largest_contour = max(contours, key=cv2.contourArea)                

                    # Seleção da área e recorte
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    crop = img[y:y+h, x:x+w]

                    # Salvar o recorte
                    if output_path != "" and os.path.isdir(output_path):
                        cv2.imwrite((output_path + "/" if output_path[-1] != "/" else output_path) + file.name, crop)
                    else:
                        cv2.imwrite("cropped_images/" + file.name, crop)

                except Exception as e:
                    print(Colors().ERROR(e))
                    continue
        
            else:
                # Arquivo não é uma imagem
                pass
            
        print(Colors().INFO("Processamento completo"))

    else:
        print("Error: Diretório não existente")