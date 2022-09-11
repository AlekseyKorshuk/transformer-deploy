import onnx, onnxruntime

model_name = '/model-storage/onnx-lit-v2-with-past/model.onnx'
onnx_model = onnx.load(model_name)

onnx.save(onnx_model, "test-model/model.onnx", save_as_external_data=True)
