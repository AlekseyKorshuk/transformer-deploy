from typing import Callable, Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import time, tqdm
import torch

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding
from transformer_deploy.benchmarks.utils import generate_multiple_inputs
from transformers import AutoModelForCausalLM

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

X = list(range(1, 512))

inputs_pytorch = generate_multiple_inputs(
    batch_size=1,
    seq_len=3,
    input_names=["input_ids"],
    device="cuda",
    nb_inputs_to_gen=10,
)

Y_onnx = []
for i in tqdm.tqdm(X):
    input_ids = torch.tensor([[1] * i], dtype=torch.int64).to(device)
    attention_mask = torch.tensor([[1] * i], dtype=torch.int64).to(device)
    start_time = time.time()
    result = infer_ort({"input_ids": input_ids, "attention_mask": attention_mask})
    # data = onnx_model(input_ids=input_ids, attention_mask=attention_mask)
    duration = time.time() - start_time
    Y_onnx.append(duration)
# print(result)

del ort_model

torch_model = AutoModelForCausalLM.from_pretrained("hakurei/litv2-6B-rev2").to(0)
Y_torch = []
with torch.no_grad():
    for i in tqdm.tqdm(X):
        input_ids = torch.tensor([[1] * i]).to(device)
        attention_mask = torch.tensor([[1] * i]).to(device)
        start_time = time.time()
        result = torch_model(**{"input_ids": input_ids})
        # data = onnx_model(input_ids=input_ids, attention_mask=attention_mask)
        duration = time.time() - start_time
        Y_torch.append(duration)
# print(result)

plt.plot(X, Y_torch, label="torch")
plt.plot(X, Y_onnx, label="onnx")
plt.legend()

plt.savefig('plot.png')
plt.show()
