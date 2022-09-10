from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from generation_mixin import GenerationMixin

config = AutoConfig.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello,", return_tensors="pt")
model = AutoModelForCausalLM.from_config(config)

del model.model
model.model = None
# mixin = GenerationMixin()
print(type(model))
output = model(**inputs)

# print(output)
