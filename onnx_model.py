from transformers import AutoModelForCausalLM, AutoTokenizer


class OnnxModel(AutoModelForCausalLM):

    # def __init__(self):
    #     pass

    def forward(self, **args):
        print(args)
        return None


tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello,", return_tensors="pt")
model = OnnxModel.from_pretrained("gpt2")

output = model(**inputs)

print(output)
