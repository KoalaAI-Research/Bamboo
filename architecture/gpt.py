import torch
import torch.nn as nn

import architecture.transformerblock as tb

# The GPT model is composed of a stack of transformer blocks, each containing a multi-head self-attention layer and a feedforward neural network.
# The model also uses positional embeddings to encode the position of each token in the input sequence.
# The final output of the model is a linear layer that predicts the next token in the sequence.
class GPTModel(nn.Module):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        self.emb_dim = config["emb_dim"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.drop_rate = config["drop_rate"]
        self.window_size = config["window_size"]  # Added window_size

        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.context_length, self.emb_dim))
        self.drop = nn.Dropout(self.drop_rate)
        self.blocks = nn.ModuleList([
            tb.TransformerBlock(
                emb_dim=self.emb_dim, 
                num_heads=self.n_heads,
                window_size=self.window_size,  # Pass window_size
                drop_rate=self.drop_rate
            ) for _ in range(self.n_layers)
        ])
        self.norm = nn.LayerNorm(self.emb_dim)
        self.fc_out = nn.Linear(self.emb_dim, self.vocab_size)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.pos_emb[:, :x.size(1), :]
        x = token_embeddings + position_embeddings
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.fc_out(x)

        return logits
    
    def printInfo(self):
        # Get our model's parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params:,}")

        # Get the number of parameters that are traininable (note: we don't use weight tying for the output layer in this model)
        total_params_gpt2 =  total_params - sum(p.numel() for p in self.out_head.parameters())
        print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

        # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
        total_size_bytes = total_params * 4

        # Convert to megabytes
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total size of the model: {total_size_mb:.2f} MB")

    # Generate text using the model, using simple greedy decoding
    def generate_simple(self, idx, max_new_tokens):
        input_length = idx.size(1)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]  # Crop context window
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # Take the last token's logit
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, eos_id=None):
        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            with torch.no_grad():
                logits = self(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx