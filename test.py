from typing import Dict, Optional

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import pipeline, AutoTokenizer
import torch
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions

# model = ORTModelForCausalLM.from_pretrained(save_directory, file_name="model-quantized.onnx")
from transformer_deploy.backends.ort_utils import create_model_for_provider, torch_to_numpy_dtype_dict, to_pytorch

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


def inference_onnx_binding(
        model_onnx: InferenceSession,
        inputs: Dict[str, torch.Tensor],
        device: str,
        device_id: int = 0,
        binding: Optional[IOBinding] = None,
        clone_tensor: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Performs inference on ONNX Runtime in an optimized way.
    In particular, it avoids any Onnx Runtime output tensor copy.
    It means that Onnx Runtime is still owner of the array, and it will overwrite its content if you do another
    inference. To avoid any issue, just set clone_tensor to True (default).
    For best performance and lowest memory footprint, if you know what you are doing, set clone_tensor to False.

    :param model_onnx: ONNX model
    :param inputs: input torch tensor
    :param device: where to run the inference. One of [cpu, cuda]
    :param device_id: ID of the device where to run the inference, to be used when there are multiple GPUs, etc.
    :param binding: previously generated binding IO, will be reset.
    :param clone_tensor: clone Pytorch tensor to avoid its content being overwritten by Onnx Runtime
        at the next inference call.
    :return: a dict {axis name: output tensor}
    """
    assert isinstance(device, str)
    assert device in ["cpu", "cuda"], f"unexpected inference device: '{device}'"
    if binding is None:
        binding: IOBinding = model_onnx.io_binding()
    else:
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
    for input_onnx in model_onnx.get_inputs():
        if input_onnx.name not in inputs:  # some inputs may be optional
            continue
        tensor: torch.Tensor = inputs[input_onnx.name]
        tensor = tensor.detach()
        # if tensor.dtype in [torch.int64, torch.long]:
        #     # int32 mandatory as input of bindings, int64 not supported
        #     tensor = tensor.type(dtype=torch.int32)
        tensor = tensor.contiguous()
        binding.bind_input(
            name=input_onnx.name,
            device_type=device,
            device_id=device_id,
            element_type=torch_to_numpy_dtype_dict[tensor.dtype],
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
        inputs[input_onnx.name] = tensor

    for out in model_onnx.get_outputs():
        binding.bind_output(
            name=out.name,
            device_type=device,
            device_id=device_id,
        )
    binding.synchronize_inputs()
    model_onnx.run_with_iobinding(binding)
    binding.synchronize_outputs()
    outputs = dict()
    assert len(model_onnx.get_outputs()) == len(
        binding.get_outputs()
    ), f"{len(model_onnx.get_outputs())} != {len(binding.get_outputs())}"
    for out, t in zip(model_onnx.get_outputs(), binding.get_outputs()):
        outputs[out.name] = to_pytorch(t, clone_tensor=clone_tensor)
    return outputs


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
result = model.generate(**inputs)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
# result = pipe("Hello,")
print(result)
