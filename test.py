import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation_utils import GenerationMixin
import numpy as np
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from onnxruntime import InferenceSession
from typing import Dict
from transformer_deploy.backends.ort_utils import create_model_for_provider, torch_to_numpy_dtype_dict, to_pytorch, \
    inference_onnx_binding
from datasets import load_dataset
import tqdm
import time
import matplotlib.pyplot as plt

GENERATION_KWARGS = {
    "max_new_tokens": 64,
    # "min_new_tokens": 8,
    'eos_token_id': 198,
    'do_sample': False,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, list):
        return [to_numpy(e) for e in x]
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}


def to_pt(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.tensor(x).to(0)
    if isinstance(x, list):
        return [to_pt(e) for e in x]
    if isinstance(x, dict):
        return {k: to_pt(v) for k, v in x.items()}


class ONNXWrapper(GenerationMixin):
    "Wrapps ONNX `InferenceSession` to enable generation using GenerationMixin"

    def __init__(self, onnx_model, config, device="cuda") -> None:
        self.session = create_model_for_provider(
            onnx_model,
            provider_to_use="CUDAExecutionProvider",
            nb_threads=1
        )
        self.config = config
        self.main_input_name = "input_ids"
        self.device = torch.device(device)
        # TODO: check config
        self.output_names = ["logits"] + [f"present.{i // 2}.{'key' if i % 2 == 0 else 'value'}" for i in
                                          range(self.config.n_layer * 2)]
        self.past_keys = [f"past_key_values.{i // 2}.{'key' if i % 2 == 0 else 'value'}" for i in
                          range(config.n_layer * 2)]

    def forward(input_ids, attention_mask, past_key_values):
        # dummy method for compatibility
        pass

    def get_empty_past(self, input_ids, config):
        batch_size = input_ids.shape[0]
        past_shape = [batch_size, config.n_head, 0, config.n_embd // config.n_head]
        empty_past = {k: np.empty(past_shape, np.float32) for k in self.past_keys}
        return empty_past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]
        else:
            past = self.get_empty_past(input_ids, self.config)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1:]
        else:
            position_ids = None
        # position_ids not used by ONNX model
        return {
            "inputs": {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            },
            "past": past
        }

    @staticmethod
    def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder):
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        return model_kwargs

    def __call__(self, inputs, past=None, **kwargs) -> CausalLMOutputWithPast:

        inputs["attention_mask"] = inputs["attention_mask"]  # .float()

        element = past[list(past.keys())[0]]

        print("######")
        if element.shape[2] != 0:
            start_time = time.time()
            outputs = inference_onnx_binding(
                model_onnx=self.session,
                inputs={**inputs, **to_pt(past)},
                device="cuda",
                output_names=self.output_names
            )
            print(f"inference_onnx_binding: {time.time() - start_time}")
            logits = outputs.pop("logits")
            past_key_values = {k: v for k, v in zip(self.past_keys, outputs.values())}
        if True:
            # print("-------")
            start_time = time.time()
            outputs = to_pt(
                self.session.run(output_names=self.output_names, input_feed={**to_numpy(inputs), **to_numpy(past)}))
            print(f"self.session.run: {time.time() - start_time}")
            logits = outputs[0]
            past_key_values = {k: v for k, v in zip(self.past_keys, outputs[1:])}


        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values
        )


model_id = "hakurei/litv2-6B-rev2"
config = AutoConfig.from_pretrained(model_id)
model = ONNXWrapper("onnx-lit-with-past/model.onnx", config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("ChaiML/user_model_inputs")

X = dataset["train"]["text"][:10]

Y_onnx = []
onnx_outputs = []
for i in tqdm.tqdm(X):
    inputs = tokenizer(i, return_tensors="pt").to(0)
    start_time = time.time()
    output = model.generate(**inputs, **GENERATION_KWARGS)
    duration = time.time() - start_time
    Y_onnx.append(duration)
    onnx_outputs.append(tokenizer.decode(output[0])[len(i):])

del model

torch_model = AutoModelForCausalLM.from_pretrained("hakurei/litv2-6B-rev2").to(0)
Y_torch = []
torch_outputs = []
with torch.no_grad():
    for i in tqdm.tqdm(X):
        inputs = tokenizer(i, return_tensors="pt").to(0)
        start_time = time.time()
        output = torch_model.generate(**inputs, **GENERATION_KWARGS)
        duration = time.time() - start_time
        Y_torch.append(duration)
        torch_outputs.append(tokenizer.decode(output[0])[len(i):])

# print(result)
plt.plot(list(range(len(X))), Y_torch, label="torch")
plt.plot(list(range(len(X))), Y_onnx, label="onnx")
plt.legend()

plt.savefig('plot.png')
plt.show()

for torch, onnx in zip(torch_outputs, onnx_outputs):
    print(torch)
    print("-" * 100)
    print(onnx)
    print("#" * 100)
