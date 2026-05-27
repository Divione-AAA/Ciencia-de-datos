from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Cargar el modelo entrenado
model = load_model("../best_emotion_model.keras")

# Definir las clases de emociones (ajusta según tu modelo)
EMOTIONS = []#Definir despues

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('L')  # Escala de grises
        img = img.resize((48, 48))  # Ajusta el tamaño según tu modelo
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        preds = model.predict(img_array)
        emotion_idx = np.argmax(preds[0])
        emotion = EMOTIONS[emotion_idx]
        return {"emotion": emotion, "confidence": float(np.max(preds[0]))}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
