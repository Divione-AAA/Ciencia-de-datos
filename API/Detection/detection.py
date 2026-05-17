from fastapi import APIRouter, FastAPI, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np

emo_dect = APIRouter()
@emo_dect.post("/")
def upload(imagen: UploadFile):

    imagen = Image.open(BytesIO(imagen.file.read()))
    imagen = np.array(imagen)
    return Detection(imagen)