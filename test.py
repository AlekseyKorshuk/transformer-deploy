from typing import Dict, Optional

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import pipeline, AutoTokenizer
import torch
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions

# model = ORTModelForCausalLM.from_pretrained(save_directory, file_name="model-quantized.onnx")
from transformer_deploy.backends.ort_utils import create_model_for_provider, torch_to_numpy_dtype_dict, to_pytorch, inference_onnx_binding

GENERATION_KWARGS = {
    "max_new_tokens": 64,
    'eos_token_id': 198,
    'do_sample': False,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = ORTModelForCausalLM.from_pretrained("gpt2", from_transformers=True)
model.to(torch.device("cuda"))

input_ids = torch.tensor([[1] * 10] * 1, dtype=torch.int64).to(0)
attention_mask = torch.tensor([[1] * 10] * 1, dtype=torch.int64).to(0)

output1 = model.model.run(None, {"input_ids": input_ids.cpu().detach().numpy(),
                                 "attention_mask": attention_mask.cpu().detach().numpy()})
print(output1)
print("#" * 100)

input_ids = torch.tensor([[1] * 10] * 1, dtype=torch.int64).to(0)
attention_mask = torch.tensor([[1] * 10] * 1, dtype=torch.int64).to(0)

output1 = model(input_ids=input_ids, attention_mask=attention_mask)
print(output1)
print("#" * 100)




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

output2 = engine.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
print(output2)
print("#" * 100)

model.model = engine

result = model(**{"input_ids": input_ids, "attention_mask": attention_mask})

print(result)
print("#" * 100)

inputs = tokenizer("Hello,", return_tensors="pt").to(0)
result = model.generate(**inputs, **GENERATION_KWARGS)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
# result = pipe("Hello,")
print(result)
