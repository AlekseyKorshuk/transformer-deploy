from inference_service import ONNXWrapper
import os
from transformers import AutoConfig, AutoTokenizer
import time

MODEL_ID = "hakurei/litv2-6B-rev2"
MODEL_PATH = "/model-storage/onnx-lit-v2-with-past-new"
MODEL_FILENAME = "model_fp16.onnx"

print("RUNNING TEST")
start_time = time.time()
config = AutoConfig.from_pretrained(MODEL_ID)
model = ONNXWrapper(os.path.join(MODEL_PATH, MODEL_FILENAME), config)
print(f"Model loaded in {time.time() - start_time} seconds")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
inputs = tokenizer("Test", return_tensors="pt").to(0)
outputs = model.generate(**inputs)
print(outputs)
input("Enter to exit")
