from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from openvino.runtime import Core
import time
CLASSES = ["Bee", "Ant"]

ie = Core()
app = FastAPI()

model = ie.read_model(model="model/saved_model.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.output(0)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((224, 224)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    result_infer = compiled_model([img_batch])[output_layer]
    result_index = np.argmax(result_infer[0])
    
    # index = np.argmax(predictions[0])
    class_name = CLASSES[result_index]

    # predicted_class = class_names[np.argmax(predictions[0])]
    # confidence = np.max(result_index[0])
    # confidence = round(confidence, 4)
    return {"class": class_name}
    # return {"class": class_name, "confidence": float(round(confidence, 4))}
