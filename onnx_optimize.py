import os
from transformer_deploy.backends.ort_utils import optimize_onnx
from transformer_deploy.backends.pytorch_utils import get_model_size
from transformers import AutoConfig, PretrainedConfig

model_name = "hakurei/litv2-6B-rev2"
onnx_model_path = os.path.join("./onnx-lit", "model.onnx")
onnx_optim_model_path = os.path.join("./onnx-lit", "model-optimized.onnx")
# model_config: PretrainedConfig = AutoConfig.from_pretrained(
#     pretrained_model_name_or_path=model
# )
#
# num_attention_heads = 16
# hidden_size = 4096
# # num_attention_heads, hidden_size = get_model_size(path=model)
#
# optimize_onnx(
#     onnx_path=onnx_model_path,
#     onnx_optim_model_path=onnx_optim_model_path,
#     fp16=True,
#     use_cuda=True,
#     num_attention_heads=num_attention_heads,
#     hidden_size=hidden_size,
#     architecture=model_config.model_type,
# )

from optimum.onnxruntime import ORTModelForCausalLM

model = ORTModelForCausalLM.from_pretrained("./onnx-lit", file_name="model.onnx")
# model = ORTModelForCausalLM.from_pretrained("gpt2", from_transformers=True)

from optimum.onnxruntime.configuration import OptimizationConfig

# Here the optimization level is selected to be 1, enabling basic optimizations such as redundant
# node eliminations and constant folding. Higher optimization level will result in a hardware
# dependent optimized graph.
optimization_config = OptimizationConfig(
    optimization_level=0,
    optimize_for_gpu=True,
    enable_gelu_approximation=False,
    fp16=True
)

from optimum.onnxruntime import ORTOptimizer

optimizer = ORTOptimizer.from_pretrained(
    model
)

# Export the optimized model
optimizer.export(
    onnx_model_path=onnx_model_path,
    onnx_optimized_model_output_path=onnx_optim_model_path,
    optimization_config=optimization_config,
)
