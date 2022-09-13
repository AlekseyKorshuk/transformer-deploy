import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
import tqdm
from datasets import load_dataset
import warnings

dataset = load_dataset("ChaiML/user_model_inputs")
model_id = "hakurei/litv2-6B-rev2"

GENERATION_KWARGS = {
    "max_new_tokens": 32,
    'eos_token_id': 198,
    'do_sample': True,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}
model = AutoModelForCausalLM.from_pretrained(model_id).half().eval().to(0)

INPUT_EXAMPLES = dataset["train"]["text"][:10]

# load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

accelerated_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Pytorch single batch"):
    inputs = tokenizer(example, return_tensors='pt').to(0)
    result = model.generate(**inputs, **GENERATION_KWARGS)

model.to("cpu")
ds_model = deepspeed.init_inference(
    model=model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float16,  # dtype of the weights (fp16)
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

accelerated_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Accelerated single batch"):
    inputs = tokenizer(example, return_tensors='pt').to(0)
    result = ds_model.generate(**inputs, **GENERATION_KWARGS)
