from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import load_model, save_model

onnx_model = load_model('/model-storage/onnx-lit-v2-with-past/model.onnx')
new_onnx_model = convert_float_to_float16(onnx_model)
save_model(new_onnx_model, '/model-storage/onnx-lit-v2-with-past-fp16/model.onnx')