# API de Detección de Expresiones Humanas

Esta API utiliza FastAPI y un modelo entrenado para detectar la expresión facial en una imagen enviada por el usuario.

## Uso

1. Inicia el servidor:

   ```bash
   uvicorn main:app --reload
   ```

2. Envía una imagen a la ruta `/predict/` usando un cliente HTTP (por ejemplo, Postman o curl):

   - Endpoint: `POST /predict/`
   - Formato: multipart/form-data
   - Campo: `file` (imagen a analizar)

3. Respuesta:

   ```json
   {
     "emotion": "Felicidad",
     "confidence": 0.98
   }
   ```

## Notas
- El modelo debe estar en la ruta `../best_emotion_model.keras` respecto a este archivo.
- Las clases de emociones pueden ajustarse en el código según el modelo entrenado.
