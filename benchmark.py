from typing import Callable, Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import time, tqdm
import torch

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding
from transformer_deploy.benchmarks.utils import generate_multiple_inputs
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from inference_service import ONNXWrapper
import os
from transformers import AutoConfig, AutoTokenizer
import time

from transformer_deploy.convert import check_accuracy

dataset = load_dataset("ChaiML/user_model_inputs")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

MODEL_ID = "hakurei/litv2-6B-rev2"
MODEL_PATH = "/model-storage/onnx-lit-v2-with-past-new"
FP16_MODEL_FILENAME = "model_fp16.onnx"
MODEL_FILENAME = "model.onnx"


config = AutoConfig.from_pretrained(MODEL_ID)
fp16_model = ONNXWrapper(os.path.join(MODEL_PATH, FP16_MODEL_FILENAME), config)


device = 0
batch_size = 1
X = dataset["train"]["text"][:10]

Y_onnx = []
onnx_outputs = []
for i in tqdm.tqdm(X):
    # input_ids = torch.tensor([[i] * i] * batch_size, dtype=torch.int64).to(device)
    # attention_mask = torch.tensor([[i] * i] * batch_size, dtype=torch.int64).to(device)
    inputs = tokenizer(i, return_tensors="pt").to(0)
    start_time = time.time()
    result = fp16_model(**inputs)
    # data = onnx_model(input_ids=input_ids, attention_mask=attention_mask)
    duration = time.time() - start_time
    Y_onnx.append(duration)
    # onnx_outputs.append(result)
# print(result)
# print(result.size())

del fp16_model

torch_model = AutoModelForCausalLM.from_pretrained("hakurei/litv2-6B-rev2").to(0)
Y_torch = []
torch_outputs = []
with torch.no_grad():
    for i in tqdm.tqdm(X):
        # input_ids = torch.tensor([[i] * i] * batch_size).to(device)
        # attention_mask = torch.tensor([[i] * i] * batch_size).to(device)
        inputs = tokenizer(i, return_tensors="pt").to(0)
        start_time = time.time()
        result = torch_model.generate(**inputs)
        duration = time.time() - start_time
        Y_torch.append(duration)
        # torch_outputs.append(result)
# print(result)
# print(result.size())

plt.plot(list(range(len(X))), Y_torch, label="torch")
plt.plot(list(range(len(X))), Y_onnx, label="onnx")
plt.legend()

plt.savefig('plot.png')
plt.show()
