from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import onnxruntime
import numpy as np
from typing import List, Union
from PIL import Image

MODEL_PATH = "./model.onnx"

try:
    session = onnxruntime.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape

    print(f"Model loaded. Input: {input_name}, Shape: {input_shape}, Output: {output_name}")


except Exception as e:
    raise Exception(f"Failed to load ONNX model: {e}")

app = FastAPI(title="Inference", description="Cats and dogs classification")

class InferenceResponse(BaseModel):
    predictions: List[List[float]]

def preprocess_image(image_file: UploadFile) -> np.ndarray:
    try:
        with image_file.file as file:
            img = Image.open(file).convert("RGB")
        img = img.resize((input_shape[2], input_shape[3]), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array.transpose((2, 0, 1))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(3):
            img_array[i] = (img_array[i] - mean[i]) / std[i]
        return img_array
    except Exception as e:
        if isinstance(e, Image.UnidentifiedImageError):
            raise ValueError(f"Invalid image file: {e}") from e
        else:
            raise ValueError(f"Preprocessing error: {e}") from e

def postprocess_output(output: np.ndarray) -> List[List[float]]:
    exp_output = np.exp(output - np.max(output, axis=-1, keepdims=True))
    softmax_output = exp_output / np.sum(exp_output, axis=-1, keepdims=True)
    return softmax_output.tolist()

@app.post("/predict", response_model=InferenceResponse)
async def predict(image: UploadFile = File(...)):
    try:
        input_data = preprocess_image(image)
        input_data = np.expand_dims(input_data, axis=0) 

        raw_result = session.run([output_name], {input_name: input_data})[0]
        predictions = postprocess_output(raw_result)
        return InferenceResponse(predictions=predictions)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.get("/")
async def root():
    return {"message": "Inference Server is running!"}