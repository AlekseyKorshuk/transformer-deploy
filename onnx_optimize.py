import os
from transformer_deploy.backends.ort_utils import optimize_onnx
from transformer_deploy.backends.pytorch_utils import get_model_size
from transformers import AutoConfig, PretrainedConfig

model = "hakurei/litv2-6B-rev2"
onnx_model_path = os.path.join("./onnx-lit", "model.onnx")
onnx_optim_model_path = os.path.join("./onnx-lit", "model-optimized.onnx")
model_config: PretrainedConfig = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model
)

num_attention_heads, hidden_size = get_model_size(path=model)

optimize_onnx(
    onnx_path=onnx_model_path,
    onnx_optim_model_path=onnx_optim_model_path,
    fp16=True,
    use_cuda=True,
    num_attention_heads=num_attention_heads,
    hidden_size=hidden_size,
    architecture=model_config.model_type,
)
