import gc
import kfserving
import logging
import math
import os
import signal
import sys

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation_utils import GenerationMixin

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

# noinspection PyUnresolvedReferences
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions
import cupy as cp
import ctypes as C
from ctypes.util import find_library
import time
import tqdm
import subprocess

libc = C.CDLL(find_library("c"))
libc.malloc.restype = C.c_void_p

SERVER_NUM_WORKERS = int(os.environ.get('SERVER_NUM_WORKERS', 1))
SERVER_PORT = int(os.environ.get('SERVER_PORT', 8080))
MODEL_DEVICE = 0
MODEL_PATH = os.environ.get('MODEL_PATH', '/mnt/models')
MODEL_NAME = os.environ.get('MODEL_NAME', 'GPT-J-6B-lit-v2')
MODEL_FILENAME = os.environ.get('MODEL_FILENAME', 'model.onnx')
MODEL_PRECISION = os.environ.get('MODEL_PRECISION', 'native').lower()
READY_FLAG = '/tmp/ready'
DEBUG_MODE = bool(os.environ.get('DEBUG_MODE', 0))
MODEL_ID = os.environ.get('MODEL_ID', 'hakurei/litv2-6B-rev2')

subprocess.run("mkdir ./model", shell=True)
subprocess.run(f"cp -r {MODEL_PATH}/ ./model/", shell=True)
subprocess.run(f"ls ./model", shell=True)

MODEL_PATH = "./model/models"

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)
logger = logging.getLogger(MODEL_NAME)


def create_model_for_provider(
        path: str,
        provider_to_use: Union[str, List],
        nb_threads: int = 0,
        optimization_level: GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        enable_profiling: bool = False,
        log_severity: int = 2,
) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = optimization_level
    options.enable_profiling = enable_profiling
    options.log_severity_level = log_severity
    if isinstance(provider_to_use, str):
        provider_to_use = [provider_to_use]
    if provider_to_use == ["CPUExecutionProvider"]:
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = nb_threads
    return InferenceSession(path, options, providers=provider_to_use)


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


# https://github.com/pytorch/pytorch/blob/ac79c874cefee2f8bc1605eed9a924d80c0b3542/torch/testing/_internal/common_utils.py#L349
numpy_to_torch_dtype_dict: Dict[np.dtype, torch.dtype] = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

torch_to_numpy_dtype_dict: Dict[torch.dtype, np.dtype] = {v: k for k, v in numpy_to_torch_dtype_dict.items()}

# np.ctypeslib.as_ctypes_type not used as it imply to manage exception, etc. for unsupported types like float16
ort_conversion_table: Dict[str, Tuple[torch.dtype, Optional[np.dtype], Optional[int], int]] = {
    # bool not supported by DlPack! see https://github.com/dmlc/dlpack/issues/75
    "tensor(bool)": (torch.bool, None, C.c_bool, 1),
    "tensor(int8)": (torch.int8, np.int8, C.c_int8, 1),
    "tensor(int16)": (torch.int16, np.int16, C.c_int16, 2),
    "tensor(int32)": (torch.int32, np.int32, C.c_int32, 4),
    "tensor(int64)": (torch.int64, np.int64, C.c_int64, 8),
    "tensor(float16)": (torch.float16, np.float16, None, 2),
    "tensor(bfloat16)": (torch.bfloat16, None, None, 2),  # bfloat16 not supported by DlPack!
    "tensor(float)": (torch.float32, np.float32, C.c_float, 4),
    "tensor(double)": (torch.float64, np.float64, C.c_double, 8),
}


def to_pytorch(ort_tensor: OrtValue, clone_tensor: bool) -> torch.Tensor:
    """
    Convert OrtValue output by Onnx Runtime to Pytorch tensor.
    The process can be done in a zero copy way (depending of clone parameter).
    :param ort_tensor: output from Onnx Runtime
    :param clone_tensor: Onnx Runtime owns the storage array and will write on the next inference.
        By cloning you guarantee that the data won't change.
    :return: Pytorch tensor
    """
    ort_type = ort_tensor.data_type()
    torch_type, np_type, c_type, element_size = ort_conversion_table[ort_type]
    use_cuda = ort_tensor.device_name().lower() == "cuda"
    use_intermediate_dtype = np_type is None or (c_type is None and not use_cuda)
    # some types are not supported by numpy (like bfloat16), so we use intermediate dtype
    # same for ctype if tensor is on CPU
    if use_intermediate_dtype:
        np_type = np.byte
        # fake type as some don't exist in C, like float16
        c_type = C.c_byte
        nb_elements = np.prod(ort_tensor.shape())
        data_size = nb_elements * element_size
        input_shape = (data_size,)
    else:
        input_shape = ort_tensor.shape()
    if use_cuda:
        fake_owner = 1
        # size not used anywhere, so just put 0
        memory = cp.cuda.UnownedMemory(ort_tensor.data_ptr(), 0, fake_owner)
        memory_ptr = cp.cuda.MemoryPointer(memory, 0)
        # make sure interpret the array shape/dtype/strides correctly
        cp_array = cp.ndarray(shape=input_shape, memptr=memory_ptr, dtype=np_type)
        # cloning required otherwise ORT will recycle the storage array and put new values into it if new inf is done.
        torch_tensor: torch.Tensor = torch.from_dlpack(cp_array.toDlpack())
    else:
        data_pointer = C.cast(ort_tensor.data_ptr(), C.POINTER(c_type))
        np_tensor = np.ctypeslib.as_array(data_pointer, shape=input_shape)
        torch_tensor = torch.from_numpy(np_tensor)
    # convert back to the right type
    if use_intermediate_dtype:
        # https://github.com/csarofeen/pytorch/pull/1481 -> no casting, just reinterpret_cast
        torch_tensor = torch_tensor.view(torch_type)
        torch_tensor = torch_tensor.reshape(ort_tensor.shape())
    if clone_tensor:
        torch_tensor = torch_tensor.clone()
    return torch_tensor


def inference_onnx_binding(
        model_onnx: InferenceSession,
        inputs: Dict[str, torch.Tensor],
        device: str,
        device_id: int = 0,
        binding: Optional[IOBinding] = None,
        clone_tensor: bool = True,
        output_names=None
) -> Dict[str, torch.Tensor]:
    assert isinstance(device, str)
    assert device in ["cpu", "cuda"], f"unexpected inference device: '{device}'"
    if output_names is None:
        output_names = [out.name for out in model_onnx.get_outputs()]
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
        # tensor = tensor.type(dtype=torch.int64)
        tensor = tensor.contiguous()
        # print(f"pointer to {input_onnx.name}: {tensor.data_ptr()}")
        print(tensor.dtype)
        print(torch_to_numpy_dtype_dict[tensor.dtype])
        if tensor.data_ptr() == 0:
            ortvalue = OrtValue.ortvalue_from_shape_and_type(
                tensor.shape,
                np.float16,
                'cuda', 0
            )
            binding.bind_ortvalue_input(input_onnx.name, ortvalue)
        else:
            binding.bind_input(
                name=input_onnx.name,
                device_type=device,
                device_id=device_id,
                element_type=torch_to_numpy_dtype_dict[tensor.dtype],
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )
        inputs[input_onnx.name] = tensor

    for out in output_names:
        binding.bind_output(
            name=out,
            device_type=device,
            device_id=device_id,
        )
    binding.synchronize_inputs()
    model_onnx.run_with_iobinding(binding)
    binding.synchronize_outputs()
    outputs = dict()
    assert len(output_names) == len(
        binding.get_outputs()
    ), f"{len(output_names)} != {len(binding.get_outputs())}"
    for out, t in zip(output_names, binding.get_outputs()):
        outputs[out] = to_pytorch(t, clone_tensor=clone_tensor)
    return outputs


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

        outputs = inference_onnx_binding(
            model_onnx=self.session,
            inputs={**inputs, **to_pt(past)},
            device="cuda",
            output_names=self.output_names
        )
        logits = outputs.pop("logits")
        past_key_values = {k: v for k, v in zip(self.past_keys, outputs.values())}

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values
        )


class KFServingHuggingFace(kfserving.KFModel):
    def __init__(self, name):
        super().__init__(name)
        self.name = MODEL_NAME
        self.ready = False
        self.config = None
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.bad_words_ids = None

    def load_config(self):
        logger.info(f'Loading config from {MODEL_ID}')
        self.config = AutoConfig.from_pretrained(MODEL_ID)
        logger.info('Config loaded.')

    def load_tokenizer(self):
        logger.info(f'Loading tokenizer from {MODEL_ID} ...')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.pad_token_id = ['<|endoftext|>']
        assert self.tokenizer.pad_token_id == 50256, 'incorrect padding token'
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        logger.info('Tokenizer loaded.')

    def load_bad_word_ids(self):
        logger.info('loading bad word ids')

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            add_prefix_space=True
        )

        forbidden = [
            'nigger', 'nigga', 'negro', 'blacks',
            'rapist', 'rape', 'raping', 'niggas', 'raper',
            'niggers', 'rapers', 'niggas', 'NOOOOOOOO',
            'fag', 'faggot', 'fags', 'faggots']

        bad_words_ids = []
        for word in forbidden:
            bad_words_ids.append(tokenizer(word).input_ids)
        self.bad_words_ids = bad_words_ids

        logger.info('done loading bad word ids')

    def load(self):
        """
        Load from a pytorch saved pickle to reduce the time it takes
        to load the model.  To benefit from this it is important to
        have run pytorch save on the same machine / hardware.
        """

        gc.disable()
        start_time = time.time()

        self.load_config()
        self.load_tokenizer()
        self.load_bad_word_ids()

        logger.info(
            f'Loading model from {MODEL_PATH} into device {MODEL_DEVICE}:{torch.cuda.get_device_name(MODEL_DEVICE)}')

        config = AutoConfig.from_pretrained(MODEL_ID)
        self.model = ONNXWrapper(os.path.join(MODEL_PATH, MODEL_FILENAME), config)

        self.model.config.eos_token_id = 198
        self.model.config.exponential_decay_length_penalty = None
        self.model.eos_token_id = 198
        logger.info('Model loaded.')

        logger.info('Warming up...')
        self.warmup()

        logger.info('Creating generator for model ...')
        logger.info(f'Model is ready in {str(time.time() - start_time)} seconds.')

        gc.enable()
        self.ready = True
        self._set_ready_flag()

    def explain(self, request):
        text = request['input_text']
        output_text = request['output_text']
        args = request['parameters']

        input_tokens = self.tokenizer(text, return_tensors="pt")['input_ids'].to(0)
        output_tokens = self.tokenizer(output_text, return_tensors="pt")['input_ids'].to(0)

        args['return_dict_in_generate'] = True
        args['output_scores'] = True
        args['max_new_tokens'] = 1

        logprobs = []
        tokens = []
        for token in output_tokens[0]:
            output = self.model.generate(input_tokens, **args)
            output_probs = torch.stack(output.scores, dim=1).softmax(-1)[0][0]
            prob = output_probs[token]
            logprobs.append(math.log(prob) if prob > 0 else -9999)
            tokens.append(self.tokenizer.decode(token))
            input_tokens = torch.cat((input_tokens, token.resize(1, 1)), dim=1)
        return {'tokens': tokens, 'logprobs': logprobs}

    def predict(self, request, parameters=None):
        # batching requires fixed parameters
        request_params = {
            'temperature': 0.72,
            'repetition_penalty': 1.13125,
            'max_new_tokens': 64,
            'top_p': 0.725,
            'top_k': 0,
            'do_sample': True,
            'eos_token_id': 198,
            'bad_words_ids': self.bad_words_ids
        }

        if parameters is not None:
            request_params.update(parameters)

        inputs = request['instances']

        input_ids = self.tokenizer(
            inputs,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True).to(0)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids['input_ids'],
                attention_mask=input_ids['attention_mask'],
                **request_params)

        responses = []
        for ins, outs in zip(inputs, outputs):
            decoded = self.tokenizer.decode(
                outs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            decoded = decoded[len(ins):]
            responses.append(decoded.rstrip())

        return {'predictions': responses}

    def warmup(self):
        for i in tqdm.trange(10, desc="Warming up"):
            self.predict(
                request={
                    "instances": [
                        "ear's Fright and tried to kill the night guard, who is Michael, Henry or a random unnamed person. Eventually, the attraction is caught on fire. In the newspaper, Springtrap's head can be seen when brightening up the image, giving an early hint he survived.\n\nIn the opening scene of Sister Location, an entrepreneur is asking him questions about the new animatronics. They inquire why certain features were added and express their concerns, but he avoids answering the specific features they refer to.\n\nHe is also the creator of the Funtime Animatronics (Assisted by an unknowing Henry) and the former owner of the Circus Baby's Entertainment and Rental, and, by extension, Circus Baby's Pizza World.\n\nIt's revealed in the final Michael Afton's Cutscene that William sent his son, Michael, to his rundown factory to find his daughter, but he is 'scooped' as his sister, Baby, tricked him. Ennard took control over his body, but he manages to survive as Michael becomes a rotting corpse. He swears to find him.\n\nWilliam Afton returns as the main antagonist. It's revealed that William's old partner, Henry, lured Springtrap, Scrap Baby (Elizabeth), Molten Freddy (and by extension, the remaining parts of Ennard), and Lefty (the Puppet) to a new Freddy Fazbear's Pizza. Michael in Freddy Fazbear's Pizzeria Simulator is the manager. On Saturday, Henry burns the whole pizzeria down, while he dies in the fire. Michael decides to stay in the fire as well. Springtrap and every other animatronic die in the fire and the souls are free, as their killer is dead.\n\nWhile not directly appearing, footprints that are very similar to Springtrap's can be found behind the house in Midnight Motorist's secret minigame, presumably luring away the child of the abusive father in the game.\n\nSeen when completing the Fruity Maze game, standing next to a girl named Susie from the right is William Afton wearing the Spring Bonnie suit that he eventually was trapped in and became Springtrap he then seemingly murders Susie.\nWilliam Afton: ...\nMe: \u2026\nWilliam Afton:"]

                }
            )

    def _set_ready_flag(self):
        """Used by readiness probe. """
        with open(READY_FLAG, 'w') as fh:
            fh.write('1')


def terminate(signal, frame):
    """
    Kubernetes send SIGTERM to containers for them
    to stop so this must be handled by the process.
    """
    logger.info("Start Terminating")
    if os.path.exists(READY_FLAG):
        os.remove(READY_FLAG)
    time.sleep(5)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate)

    if DEBUG_MODE:
        import time

        time.sleep(3600 * 10)

    model = KFServingHuggingFace(MODEL_NAME)
    model.load()

    kfserving.KFServer(
        http_port=SERVER_PORT,
        workers=SERVER_NUM_WORKERS
    ).start([model])
