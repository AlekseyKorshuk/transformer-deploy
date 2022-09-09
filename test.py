from optimum.onnxruntime import ORTModelForCausalLM
from transformers import pipeline, AutoTokenizer
import torch
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions

# model = ORTModelForCausalLM.from_pretrained(save_directory, file_name="model-quantized.onnx")
from transformer_deploy.backends.ort_utils import create_model_for_provider, inference_onnx_binding

model = ORTModelForCausalLM.from_pretrained("gpt2", from_transformers=True)
model.to(torch.device("cuda"))

input_ids = torch.tensor([[1] * 10] * 1, dtype=torch.int64).to(0)
attention_mask = torch.tensor([[1] * 10] * 1, dtype=torch.int64).to(0)

output1 = model.model.run(None, {"input_ids":input_ids.cpu().detach().numpy(), "attention_mask": attention_mask.cpu().detach().numpy()})

class InferenceSessionWithIOBinding(InferenceSession):
    def __init__(self, model_path, provider, nb_threads=1):
        self.ort_model = create_model_for_provider(
            path=model_path,
            provider_to_use=provider,
            nb_threads=nb_threads,
        )

    def run(self, output_names, input_feed, run_options=None):
        results = inference_onnx_binding(model_onnx=self.ort_model, inputs=input_feed, device="cuda")
        return results
