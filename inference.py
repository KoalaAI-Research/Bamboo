import json
from architecture.gpt import GPTModel
import tiktoken
import torch

# Set the model name to load:
model_name = "bamboo-1-365M-grug"
folder_name = "./output/" + model_name

# Load the model config from json:
config_file = f"{folder_name}/config.json"

# Load the model config
with open(config_file, "r") as f:
    config = json.load(f)

# Load the tokenizer
tokenizer = tiktoken.get_encoding(config["tokenizer"])

# Load the GPT model
model = GPTModel(config)
model.load_state_dict(torch.load(f"{folder_name}/model.pth"))

# Detect GPUs and compile to bf16 if needed:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using the GPU! 🔥")
else:
    print("Using the CPU! ❄️")

if device.type == "cuda":
    # Check if the GPU supports bfloat16
    if torch.cuda.get_device_capability()[0] >= 8:
        print("Converting model to bfloat16...")
        model.to(device=device, dtype=torch.bfloat16)
        model.pos_emb.to(device, dtype=torch.bfloat16)
    else:
        model.to(device)
        model.pos_emb.to(device)


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
    prompt_ids = prompt_ids.to(device)  # Move to the same device as the model

    # Generate text
    output_ids = model.generate(prompt_ids, max_new_tokens=50, temperature=1.8, top_k=50)
    decoded_text = token_ids_to_text(output_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format