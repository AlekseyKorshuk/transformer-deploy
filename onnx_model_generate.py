from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from generation_mixin import GenerationMixin
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline, AutoTokenizer
import torch
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions
from transformer_deploy.backends.ort_utils import create_model_for_provider, torch_to_numpy_dtype_dict, to_pytorch, \
    inference_onnx_binding
from datasets import load_dataset
import tqdm
import time
import matplotlib.pyplot as plt
from inference_service import ONNXWrapper
import os
from transformers import AutoConfig, AutoTokenizer
import time

GENERATION_KWARGS = {
    "max_new_tokens": 64,
    # "min_new_tokens": 8,
    'eos_token_id': 198,
    'do_sample': False,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

dataset = load_dataset("ChaiML/user_model_inputs")

X = dataset["train"]["text"][:100]

model_path = "onnx-lit/model.onnx"
provider = "CUDAExecutionProvider"
nb_threads = 1
MODEL_ID = "hakurei/litv2-6B-rev2"
MODEL_PATH = "/model-storage/onnx-lit-v2-with-past-new"
FP16_MODEL_FILENAME = "model_fp16.onnx"
MODEL_FILENAME = "model.onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

config = AutoConfig.from_pretrained(MODEL_ID)
fp16_model = ONNXWrapper(os.path.join(MODEL_PATH, MODEL_FILENAME), config)

Y_onnx = []
onnx_outputs = []
for i in tqdm.tqdm(X):
    inputs = tokenizer(i, return_tensors="pt").to(0)
    start_time = time.time()
    output = fp16_model.generate(**inputs, **GENERATION_KWARGS)
    duration = time.time() - start_time
    Y_onnx.append(duration)
    onnx_outputs.append(tokenizer.decode(output[0])[len(i):])

del fp16_model

torch_model = AutoModelForCausalLM.from_pretrained("hakurei/litv2-6B-rev2").half().eval().to(0)
Y_torch = []
torch_outputs = []
with torch.no_grad():
    for i in tqdm.tqdm(X):
        inputs = tokenizer(i, return_tensors="pt").to(0)
        start_time = time.time()
        output = torch_model.generate(**inputs, **GENERATION_KWARGS)
        duration = time.time() - start_time
        Y_torch.append(duration)
        torch_outputs.append(tokenizer.decode(output[0])[len(i):])

# print(result)
plt.plot(list(range(len(X))), Y_torch, label="torch")
plt.plot(list(range(len(X))), Y_onnx, label="onnx")
plt.legend()

plt.savefig('plot.png')
plt.show()

# for torch, onnx in zip(torch_outputs, onnx_outputs):
#     print(torch)
#     print("-" * 100)
#     print(onnx)
#     print("#" * 100)

import pandas as pd

df = pd.DataFrame(
    {
        "torch": Y_torch,
        "onnx": Y_onnx
    }
)

print(df.describe())
