from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from generation_mixin import GenerationMixin
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
import tqdm
from datasets import load_dataset
import warnings
from typing import Dict, Optional

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import pipeline, AutoTokenizer
import torch
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions

# model = ORTModelForCausalLM.from_pretrained(save_directory, file_name="model-quantized.onnx")
from transformer_deploy.backends.ort_utils import create_model_for_provider, torch_to_numpy_dtype_dict, to_pytorch, \
    inference_onnx_binding

config = AutoConfig.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello,", return_tensors="pt").to(0)
model = AutoModelForCausalLM.from_config(config).to(0)


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


model_path = "onnx-gpt2/model.onnx"
provider = "CUDAExecutionProvider"
nb_threads = 1

onnx_model = InferenceSessionWithIOBinding(model_path=model_path, provider=provider, nb_threads=nb_threads)

mixin = GenerationMixin(
    model=model,
    onnx_model=onnx_model
)
print(type(model))
output = mixin.generate(**inputs)

print(output)
