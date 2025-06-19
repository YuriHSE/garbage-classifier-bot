import onnxruntime as ort
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    img_data = np.asarray(image).astype(np.float32) / 255.0
    img_data = img_data.transpose(2, 0, 1)  # (HWC) → (CHW)
    return img_data[np.newaxis, :]  # (1, C, H, W)

def predict(image_path):
    ort_session = ort.InferenceSession("model/trashnet.onnx")
    input_name = ort_session.get_inputs()[0].name
    img = preprocess_image(image_path)
    outputs = ort_session.run(None, {input_name: img})[0]
    return int(np.argmax(outputs))
