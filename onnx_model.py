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
model = OnnxModel.from_pretrained("gpt2")

output = model(**inputs)

# print(output)
