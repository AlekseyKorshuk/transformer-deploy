from onnxmltools.utils.float16_converter import convert_float_to_float16, convert_float_to_float16_model_path
from onnxmltools.utils import load_model, save_model
import onnx

# onnx_model = load_model('model.onnx')
new_onnx_model = convert_float_to_float16_model_path('model.onnx')
print(type(new_onnx_model))

output_path = "model_fp16.onnx"
try:
    onnx.save(new_onnx_model, output_path, save_as_external_data=False)
except:
    onnx.save(new_onnx_model, output_path, save_as_external_data=False)

# save_model(new_onnx_model, 'model_fp16.onnx', use_external_data_format=True)
