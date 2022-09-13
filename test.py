from onnxmltools.utils.float16_converter import convert_float_to_float16, convert_float_to_float16_model_path
from onnxmltools.utils import load_model, save_model

# onnx_model = load_model('model.onnx')
new_onnx_model = convert_float_to_float16_model_path('model.onnx')
save_model(new_onnx_model, 'model_fp16.onnx', use_external_data_format=True)
