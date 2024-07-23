import json
from architecture.gpt import GPTModel
import tiktoken
import torch

# Load the model config from json:
config_file = "./output/config.json"

# Load the model config
with open(config_file, "r") as f:
    config = json.load(f)

# Load the tokenizer
tokenizer = tiktoken.get_tokenizer(config["tokenizer"])

# Load the GPT model
gpt_model = GPTModel(config)
gpt_model.load_state_dict(torch.load("./output/model.pth"))

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

while True:
    # Get user input
    prompt = input("Enter a prompt: ")

    if prompt == "exit":
        break

    # Tokenize the prompt
    prompt_ids = text_to_token_ids(prompt, tokenizer)

    # Generate text
    output_ids = gpt_model.generate(prompt_ids, max_new_tokens=50, temperature=1.0, top_k=50)
    decoded_text = token_ids_to_text(output_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format