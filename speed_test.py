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
        output = logits.cpu().numpy()
        return [output]


model_path = "onnx-gpt2/model.onnx"
provider = "CUDAExecutionProvider"
nb_threads = 1

engine = InferenceSessionWithIOBinding(model_path=model_path, provider=provider, nb_threads=nb_threads)

ort_model.model = engine

# create acclerated pipeline
# ds_clf = pipeline("text-generation", model=ds_model, tokenizer=tokenizer, device=0)

print("Accelerated single batch")
accelerated_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES[:20], desc="Accelerated single batch"):
    inputs = tokenizer(example, return_tensors='pt').to(0)
    result = ort_model.generate(**inputs, **GENERATION_KWARGS)

print("Accelerated batch size")
accelerated_outputs = []
try:
    for example in tqdm.tqdm(INPUT_EXAMPLES[:10], desc="Accelerated batch size"):
        inputs = tokenizer([example] * max_batch_size, return_tensors='pt').to(0)
        result = ort_model.generate(**inputs, **GENERATION_KWARGS)
except Exception as ex:
    print(ex)
# accelerated_output = ds_clf(example, **GENERATION_KWARGS)[0]["generated_text"][len(example):]
# accelerated_outputs.append(accelerated_output)

# difference = list(set(torch_outputs) - set(accelerated_outputs))
# print(len(difference))
