"""
Para rodar a aplicação, você deve estar dentro da pasta /Project/IA
"""

# FastAPI
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

# Scripts auxiliares
from imports import loadModel

# System
import os
import io
from PIL import Image

# Torch
import torch
from torchvision import transforms
import torch.nn.functional as F

# Numpy
import numpy as np

from dotenv import load_dotenv


# Loading env
if load_dotenv():
    print(".env loaded successfully.") # Apply logging
    print(os.getenv("TOMATO"))
else:
    raise("Couldn't load the .env")


# Configuração
# Application
APP = FastAPI()

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CORS
origins = [
    "http://localhost:3000", # Atualizar CORS com .env
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

print(f"Origens: {origins}")

APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"], # Change for security
)


"""
Por conta de usarmos um modelo pré-treinado, temos que então importa-lo e carregar a arquitetura.
Logo após, necessitamos introduzir os pesos próprios nos nós do modelo carregado e então o modelo estará pronto para uso.
"""
## --- Models Loading --- ##
# CNN
## Generalist (Pode ser usado no futuro)
# if  os.path.isfile(os.getenv("GENERALIST")):
#     GENERALIST, LOADED_GENERALIST = loadModel(os.getenv("GENERALIST"), device=DEVICE)
#     print("GENERALIST Model loaded successfully")

print("Começando a carregar os modelos.")
## Experts
if  os.path.isfile(os.getenv("TOMATO")) and \
    os.path.isfile(os.getenv("WHEAT")) and \
    os.path.isfile(os.getenv("SOYBEAN")):

    TOMATO, LOADED_TOMATO = loadModel(os.getenv("TOMATO"), device=DEVICE)
    print("TOMATO Model loaded successfully")

    WHEAT, LOADED_WHEAT = loadModel(os.getenv("WHEAT"), device=DEVICE)
    print("WHEAT Model loaded successfully")

    SOYBEAN, LOADED_SOYBEAN = loadModel(os.getenv("SOYBEAN"), device=DEVICE)
    print("SOYBEAN Model loaded successfully")

else:
    raise FileNotFoundError("couldn't identify some of the experts in models archive")


# Transform - Apply models.Get_Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])


# Defs
def __preprocess_image(image_file: UploadFile):
    try:
        # Tratamento Imagem
        image_data = image_file.file.read()    
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image_tensor = transform(image).unsqueeze(0)
        return image_tensor

    except Exception as e:
        print(f"Error trying to manipulate the image: {e}")


# Pode ser usado no futuro
# def __generalist_predict(image_tensor):

#     try:
#         with torch.no_grad():

#             outputs = GENERALIST(image_tensor.to(DEVICE))
#             probabilities = F.softmax(outputs, dim=1)
#             predicted_class_idx = torch.argmax(probabilities, dim=1).item()

#             confidence = probabilities[0][predicted_class_idx].item()
        
#         predicted_class = LOADED_GENERALIST["class_names"][predicted_class_idx]

#         all_probabilities = {
#             LOADED_GENERALIST["class_names"][i]: float(probabilities[0][i])
#             for i in range(len(LOADED_GENERALIST["class_names"]))
#         }
        
#         return predicted_class, round(confidence * 100, 2), all_probabilities

#     except Exception as e:
#         print(f"Error trying to predict: {e}")


def __expert_predict(image_tensor, type: str):

    try:
        match(type):
            case "Tomato": 
                with torch.no_grad():

                    outputs = TOMATO(image_tensor.to(DEVICE))
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class_idx = torch.argmax(probabilities, dim=1).item()

                    confidence = probabilities[0][predicted_class_idx].item()
                
                predicted_class = LOADED_TOMATO["class_names"][predicted_class_idx]

                all_probabilities = {
                    LOADED_TOMATO["class_names"][i]: float(probabilities[0][i])
                    for i in range(len(LOADED_TOMATO["class_names"]))
                }
                
                return predicted_class, round(confidence * 100, 2), all_probabilities
            
            case "Wheat": 
                with torch.no_grad():

                    outputs = WHEAT(image_tensor.to(DEVICE))
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class_idx = torch.argmax(probabilities, dim=1).item()

                    confidence = probabilities[0][predicted_class_idx].item()
                
                predicted_class = LOADED_WHEAT["class_names"][predicted_class_idx]

                all_probabilities = {
                    LOADED_WHEAT["class_names"][i]: float(probabilities[0][i])
                    for i in range(len(LOADED_WHEAT["class_names"]))
                }
                
                return predicted_class, round(confidence * 100, 2), all_probabilities
            
            case "Soybean": 
                with torch.no_grad():

                    outputs = SOYBEAN(image_tensor.to(DEVICE))
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class_idx = torch.argmax(probabilities, dim=1).item()

                    confidence = probabilities[0][predicted_class_idx].item()
                
                predicted_class = LOADED_SOYBEAN["class_names"][predicted_class_idx]

                all_probabilities = {
                    LOADED_SOYBEAN["class_names"][i]: float(probabilities[0][i])
                    for i in range(len(LOADED_SOYBEAN["class_names"]))
                }
                
                return predicted_class, round(confidence * 100, 2), all_probabilities

    except Exception as e:
        print(f"Error trying to predict: {e}")


# Routes
## Debug
@APP.get("/modelinfo")
def ModelInfo():
    return {
        "models": {
            "tomato": {
                "class_names": LOADED_TOMATO["class_names"],
                "num_classes": LOADED_TOMATO["num_classes"],
                "model_info": LOADED_TOMATO["model_info"],
            },
            "wheat": {
                "class_names": LOADED_WHEAT["class_names"],
                "num_classes": LOADED_WHEAT["num_classes"],
                "model_info": LOADED_WHEAT["model_info"],
            },
            "soybean": {
                "class_names": LOADED_SOYBEAN["class_names"],
                "num_classes": LOADED_SOYBEAN["num_classes"],
                "model_info": LOADED_SOYBEAN["model_info"],
            }
        },
    }

## Predict
@APP.post("/predict/")
async def Predict( c_type: str = Form(...), file: UploadFile = File(...) ):
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only images are acceptable.")

    contents = await file.read()
    if len(contents) > 8 * 1024 * 1024:
        raise HTTPException(400, "File too large.")
    
    await file.seek(0)

    image_tensor = __preprocess_image(image_file=file)

    # Prediction
    expert_prediction = __expert_predict(image_tensor, type=c_type)


    return {
        "plant_prediction": "Teste Planta", 
        "plant_confidence": "Teste Confiança",
        "expert": {
            "predict": expert_prediction[0].upper(), 
            "predict_confidence": expert_prediction[1],
        }
        }


if __name__ == '__main__':
    uvicorn.run(APP, host='0.0.0.0', port=5000)