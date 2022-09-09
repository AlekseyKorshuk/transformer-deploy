from typing import Callable, Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import time, tqdm
import torch

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding
from transformer_deploy.benchmarks.utils import generate_multiple_inputs
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from transformer_deploy.convert import check_accuracy

dataset = load_dataset("ChaiML/user_model_inputs")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model_path = "./triton_models/model.onnx"
model_path = "./onnx-lit/model.onnx"
provider = "CUDAExecutionProvider"
nb_threads = 1
ort_model = create_model_for_provider(
    path=model_path,
    provider_to_use=provider,
    nb_threads=nb_threads,
)


def infer_ort(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    results = inference_onnx_binding(model_onnx=ort_model, inputs=inputs, device="cuda")
    return results
    # return results["output"] if "output" in results else (results["start_logits"], results["end_logits"])


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
    result = infer_ort({"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask})
    # data = onnx_model(input_ids=input_ids, attention_mask=attention_mask)
    duration = time.time() - start_time
    Y_onnx.append(duration)
    onnx_outputs.append(result)
print(result)

del ort_model

torch_model = AutoModelForCausalLM.from_pretrained("hakurei/litv2-6B-rev2").to(0)
Y_torch = []
torch_outputs = []
with torch.no_grad():
    for i in tqdm.tqdm(X):
        # input_ids = torch.tensor([[i] * i] * batch_size).to(device)
        # attention_mask = torch.tensor([[i] * i] * batch_size).to(device)
        inputs = tokenizer(i, return_tensors="pt").to(0)
        start_time = time.time()
        result = torch_model(**inputs).logits
        duration = time.time() - start_time
        Y_torch.append(duration)
        torch_outputs.append(result)
print(result)

check_accuracy(
    engine_name="ONNX",
    pytorch_output=torch_outputs,
    engine_output=onnx_outputs,
    tolerance=0.3,
)

plt.plot(X, Y_torch, label="torch")
plt.plot(X, Y_onnx, label="onnx")
plt.legend()

plt.savefig('plot.png')
plt.show()
