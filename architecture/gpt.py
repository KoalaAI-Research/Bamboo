import torch
import torch.nn as nn

import architecture.transformerblock as tb

# The GPT model is composed of a stack of transformer blocks, each containing a multi-head self-attention layer and a feedforward neural network.
# The model also uses positional embeddings to encode the position of each token in the input sequence.
# The final output of the model is a linear layer that predicts the next token in the sequence.
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg # Back up the config for later use
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.context_length = cfg["context_length"]

        # Ensure window size is set if using sliding window attention
        self.window_size = cfg["window_size"]

        self.trf_blocks = nn.Sequential(
            *[tb.TransformerBlockWithSlidingWindow(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = tb.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
        self.printInfo()

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
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
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -self.context_length:]

            # Get the predictions
            with torch.no_grad():
                logits = self(idx_cond)

            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Get the idx of the vocab entry with the highest logits value
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx
    
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, top_p=None, eos_id=None):
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

            if top_p is not None:
                # Compute cumulative probabilities of sorted logits
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)

                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p

                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove = torch.cat((torch.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, :-1]), dim=-1)

                # Set the logits to -infinity for the tokens that should be removed
                logits[sorted_indices_to_remove] = float('-inf')

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