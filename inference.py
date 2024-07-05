from architecture.gpt import GPTModel
import tiktoken
import torch

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Define the GPT configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024,  # Shortened context length (orig: 1024)
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-key-value bias
}

# Load the GPT model
gpt_model = GPTModel(GPT_CONFIG_124M)
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