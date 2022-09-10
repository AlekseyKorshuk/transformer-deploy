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

GENERATION_KWARGS = {
    "max_new_tokens": 32,
    # "min_new_tokens": 8,
    'eos_token_id': 198,
    'do_sample': True,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

config = AutoConfig.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello,", return_tensors="pt").to(0)
model = AutoModelForCausalLM.from_config(config)


class InferenceSessionWithIOBinding(InferenceSession):
    def __init__(self, model_path, provider, nb_threads=1):
        self.ort_model = create_model_for_provider(
            path=model_path,
            provider_to_use=provider,
            nb_threads=nb_threads,
        )

    def run(self, output_names, input_feed, run_options=None):
        for key in input_feed.keys():
            input_feed[key] = torch.tensor(input_feed[key], device=0)
        results = inference_onnx_binding(model_onnx=self.ort_model, inputs=input_feed, device="cuda")
        logits = results["logits"]
        # output = logits.cpu().numpy()
        return logits


dataset = load_dataset("ChaiML/user_model_inputs")

X = dataset["train"]["text"][:10]

torch_model = AutoModelForCausalLM.from_pretrained("hakurei/litv2-6B-rev2").to(0)
Y_torch = []
torch_outputs = []
with torch.no_grad():
    for i in tqdm.tqdm(X):
        inputs = tokenizer(i, return_tensors="pt").to(0)
        start_time = time.time()
        result = torch_model.generate(**inputs, **GENERATION_KWARGS)
        duration = time.time() - start_time
        Y_torch.append(duration)
        torch_outputs.append(result)
del torch_model

model_path = "onnx-lit/model.onnx"
provider = "CUDAExecutionProvider"
nb_threads = 1

onnx_model = InferenceSessionWithIOBinding(model_path=model_path, provider=provider, nb_threads=nb_threads)

mixin = GenerationMixin(
    model=model,
    onnx_model=onnx_model
)

Y_onnx = []
onnx_outputs = []
for i in tqdm.tqdm(X):
    inputs = tokenizer(i, return_tensors="pt").to(0)
    start_time = time.time()
    output = mixin.generate(**inputs, **GENERATION_KWARGS)
    duration = time.time() - start_time
    Y_onnx.append(duration)
    onnx_outputs.append(output)

# print(result)
plt.plot(list(range(len(X))), Y_torch, label="torch")
plt.plot(list(range(len(X))), Y_onnx, label="onnx")
plt.legend()

plt.savefig('plot.png')
plt.show()
