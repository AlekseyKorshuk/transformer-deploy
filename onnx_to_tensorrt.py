import argparse
import gc
import logging
import os
import tqdm
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type, Union

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from transformer_deploy.backends.ort_utils import (
    cpu_quantization,
    create_model_for_provider,
    inference_onnx_binding,
    optimize_onnx,
)
from transformer_deploy.backends.pytorch_utils import (
    convert_to_onnx,
    get_model_size,
    infer_classification_pytorch,
    infer_feature_extraction_pytorch,
)
from transformer_deploy.backends.st_utils import STransformerWrapper, load_sentence_transformers
from transformer_deploy.backends.trt_utils import build_engine, save_engine, load_engine
from transformer_deploy.benchmarks.utils import (
    compare_outputs,
    generate_multiple_inputs,
    print_timings,
    setup_logging,
    to_numpy,
    track_infer_time,
)
from transformer_deploy.triton.configuration import Configuration, EngineType
from transformer_deploy.triton.configuration_decoder import ConfigurationDec
from transformer_deploy.triton.configuration_encoder import ConfigurationEnc
from transformer_deploy.triton.configuration_question_answering import ConfigurationQuestionAnswering
from transformer_deploy.triton.configuration_token_classifier import ConfigurationTokenClassifier
from transformer_deploy.utils.args import parse_args

import tensorrt as trt
from tensorrt.tensorrt import ICudaEngine, Logger, Runtime

verbose = True
tensorrt_path = os.path.join("./tensorrt", "model.plan")
onnx_model_path = os.path.join("/model-storage/onnx-lit-v2-with-past-new", "model_fp16.onnx")
workspace_size = 40000
fp16 = True
int8 = False

trt_logger: Logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
runtime: Runtime = trt.Runtime(trt_logger)
engine: ICudaEngine = build_engine(
    runtime=runtime,
    onnx_file_path=onnx_model_path,
    logger=trt_logger,
    min_shape=(1, 6),
    optimal_shape=(1, 256),
    max_shape=(1, 256),
    workspace_size=workspace_size * 1024 * 1024,
    fp16=fp16,
    int8=int8,
)
print("here")
# print(engine.__dict__)
save_engine(engine=engine, engine_file_path=tensorrt_path)

tensorrt_model: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = load_engine(
    runtime=runtime, engine_file_path=tensorrt_path
)

