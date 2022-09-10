from transformers import AutoModelForCausalLM, AutoTokenizer


class OnnxModel(AutoModelForCausalLM):

    # def __init__(self):
    #     pass

    def __call__(self, *args, **kwargs):
        print(args)
        print(kwargs)
        return None


tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello,", return_tensors="pt")
model = AutoModelForCausalLM.from_config("hakurei/litv2-6B-rev2")

print(type(model))
output = model(**inputs)

# print(output)
