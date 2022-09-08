from typing import Callable, Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import time, tqdm
import torch

from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding

model_path = "./triton_models"
provider = "CUDAExecutionProvider"
nb_threads = 1
ort_model = create_model_for_provider(
  path=model_path,
  provider_to_use=provider,
  nb_threads=nb_threads,
)


def infer_ort(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
  results = inference_onnx_binding(model_onnx=ort_model, inputs=inputs, device="cuda")
  return results["output"] if "output" in results else (results["start_logits"], results["end_logits"])

device = 0

X = list(range(1, 511))

Y_onnx = []
for i in tqdm.tqdm(X):
  input_ids = torch.tensor([[1]*i]).to(device)
  attention_mask = torch.tensor([[1]*i]).to(device)
  start_time = time.time()
  infer_ort(input_ids)
  # data = onnx_model(input_ids=input_ids, attention_mask=attention_mask)
  duration = time.time() - start_time
  Y_onnx.append(duration)

# plt.plot(X, Y_torch, label="torch")
plt.plot(X, Y_onnx, label="onnx")
plt.legend()
plt.show()