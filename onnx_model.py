from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class OnnxModel(AutoModelForCausalLM):

    # def __init__(self):
    #     pass

    def __call__(self, *args, **kwargs):
        print(args)
        print(kwargs)
        return None


config = AutoConfig.from_pretrained("hakurei/litv2-6B-rev2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello,", return_tensors="pt")
model = AutoModelForCausalLM.from_config(config)

print(type(model))
output = model(**inputs)

# print(output)
