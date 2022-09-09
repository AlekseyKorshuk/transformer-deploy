import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
import tqdm
from datasets import load_dataset
import warnings
from typing import Dict, Optional

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import pipeline, AutoTokenizer
import torch
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions

# model = ORTModelForCausalLM.from_pretrained(save_directory, file_name="model-quantized.onnx")
from transformer_deploy.backends.ort_utils import create_model_for_provider, torch_to_numpy_dtype_dict, to_pytorch


warnings.filterwarnings("ignore")
dataset = load_dataset("ChaiML/user_model_inputs")
# Model Repository on huggingface.co
model_id = "gpt2"
# model_id = "gpt2"

stats = {}

# load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# Test pipeline
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

INPUT_EXAMPLES = dataset["train"]["text"][:100]

example = INPUT_EXAMPLES[0]
model = AutoModelForCausalLM.from_pretrained(model_id).half().to(0)

max_batch_size = 4

# torch_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
print("Pytorch single batch")
torch_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES[:20], desc="Pytorch single batch"):
    inputs = tokenizer(example, return_tensors='pt').to(0)
    result = model.generate(**inputs, **GENERATION_KWARGS)
    # torch_output = torch_pipe(example, **GENERATION_KWARGS)[0]["generated_text"][len(example):]
    # torch_outputs.append(torch_output)
print("Pytorch batch size")
torch_outputs = []
try:
    for example in tqdm.tqdm(INPUT_EXAMPLES[:10], desc="Pytorch batch size"):
        inputs = tokenizer([example] * max_batch_size, return_tensors='pt').to(0)
        result = model.generate(**inputs, **GENERATION_KWARGS)
except Exception as ex:
    print(ex)
# print(torch_output)
# init deepspeed inference engine

ort_model = ORTModelForCausalLM.from_pretrained("gpt2", from_transformers=True)
ort_model.to(torch.device("cuda"))


ds_model = deepspeed.init_inference(
    model=model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

# create acclerated pipeline
# ds_clf = pipeline("text-generation", model=ds_model, tokenizer=tokenizer, device=0)

print("Accelerated single batch")
accelerated_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES[:20], desc="Accelerated single batch"):
    inputs = tokenizer(example, return_tensors='pt').to(0)
    result = ds_model.generate(**inputs, **GENERATION_KWARGS)

print("Accelerated batch size")
accelerated_outputs = []
try:
    for example in tqdm.tqdm(INPUT_EXAMPLES[:10], desc="Accelerated batch size"):
        inputs = tokenizer([example] * max_batch_size, return_tensors='pt').to(0)
        result = ds_model.generate(**inputs, **GENERATION_KWARGS)
except Exception as ex:
    print(ex)
# accelerated_output = ds_clf(example, **GENERATION_KWARGS)[0]["generated_text"][len(example):]
# accelerated_outputs.append(accelerated_output)

# difference = list(set(torch_outputs) - set(accelerated_outputs))
# print(len(difference))
