from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()
model = tf.keras.models.load_model('model_cnn.h5')


@app.get("/")
def home():
    return {"message": "Cat vs Dog Classifier API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    score = float(prediction[0][0])

    if score > 0.5:
        label = "Dog"
    else:
        label = "Cat"

    return {"label": label, "confidence": round(score if score > 0.5 else 1 - score, 4)}