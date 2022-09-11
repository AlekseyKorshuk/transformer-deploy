import onnx, onnxruntime

model_name = '/model-storage/onnx-lit-v2-with-past/model.onnx'
onnx_model = onnx.load(model_name)
onnx_model.save("test-model/model.onnx")
