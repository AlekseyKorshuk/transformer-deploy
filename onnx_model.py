from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from generation_mixin import GenerationMixin

config = AutoConfig.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello,", return_tensors="pt")
model = AutoModelForCausalLM.from_config(config).to(0)

print(model.__dict__)
input("1")
del model._modules
model._modules = None
input("1")
# mixin = GenerationMixin()
print(type(model))
output = model(**inputs)

# print(output)
