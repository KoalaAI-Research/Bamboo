# Import necessary libraries
import datetime
import json
import math
import time
import numpy as np  # Add this line to import numpy
import matplotlib
matplotlib.use('Agg')  # Add this line to use matplotlib without a display
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset

#from galore_torch import GaLoreAdamW
import os
import torch
import tiktoken

# Our classes:
from architecture.gpt import GPTModel
from architecture.dataset import CreateDataloader

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = model.generate_simple(
            idx=encoded, max_new_tokens=50
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()
    # Plot training and validation loss against epochs (primary y-axis)
    ax1.plot(epochs_seen, train_losses, label="Training loss", marker='o')
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss", marker='x')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Share the y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot to align ticks
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6, save_every=500):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of iterations in the training process
    total_training_steps = len(train_loader) * n_epochs

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        start_time = time.time()  # Track start time for each epoch
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment  
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) / (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the current learning rate

            # Calculate and backpropagate the loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Apply gradient clipping after the warmup phase to avoid exploding gradients
            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            tokens_seen += input_batch.numel()

            # Calculate and display iterations per second
            batch_time = time.time() - start_time         

            # Periodically evaluate the model on the training and validation sets
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                # Print the current losses + steps / sec
                steps_per_second = 1.0 / batch_time
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}, "
                      f"Steps/s: {steps_per_second:.2f}"
                )

            # Periodically save the model checkpoint:
            if global_step > 0 and global_step % save_every == 0:
                out_folder = "./output/"
                torch.save(model.state_dict(), f"{out_folder}/model.pth")
                torch.save(optimizer.state_dict(), f"{out_folder}/optimizer.pth")

                # Save the config as a json file:
                with open(f"{out_folder}/config.json", "w") as f:
                    json.dump(model.config, f)

                print(f"Saved model & optimizer state dict @ step: {global_step}")

        # Generate and print a sample from the model to monitor progress
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen, track_lrs

def setup_tokenizer(gpt_config, checkpoint_path):
    print("Loading tokenizer...")
    if checkpoint_path:
        config_file = f"{checkpoint_path}/config.json"
        with open(config_file, "r") as f:
            checkpoint_config = json.load(f)
        tokenizer = tiktoken.get_encoding(checkpoint_config["tokenizer"])
        gpt_config = checkpoint_config  # Use the checkpoint's config
    else:
        tokenizer = tiktoken.get_encoding(gpt_config["tokenizer"])
    
    gpt_config["vocab_size"] = tokenizer.n_vocab
    return tokenizer, gpt_config

def load_or_initialize_model(gpt_config, settings, continue_training_from, device, compile_model):
    if continue_training_from:
        print("Continuing training from a previous checkpoint...")
        model = GPTModel(gpt_config)
        checkpoint = torch.load(f"{continue_training_from}/model.pth", map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("Initializing new model...")
        model = GPTModel(gpt_config)

    # Common model setup
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        print("Converting model to bfloat16...")
        model.to(device=device, dtype=torch.bfloat16)
        model.pos_emb.to(device, dtype=torch.bfloat16)
    else:
        model.to(device)
        model.pos_emb.to(device)

    if compile_model and not continue_training_from:
        model = compile_model_based_on_os(model)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=settings["weight_decay"])
    
    if continue_training_from:
        optimizer.load_state_dict(torch.load(f"{continue_training_from}/optimizer.pth", map_location=device))

    return model, optimizer

def compile_model_based_on_os(model):
    print("Compiling model...")
    if os.name == "nt":
        print("Windows OS detected. Compiling as torch.compile() + Setting trinitron fallback to True.")
        model = torch.compile(model)
        torch._dynamo.config.suppress_errors = True
    else:
        model = thunder.jit(model)
    return model

def main(gpt_config, settings, continue_training_from="", compile_model=True):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using the {'GPU! 🔥' if device.type == 'cuda' else 'CPU! ❄️'}")

    # Old-style data loading from a local file, uncomment if you want to use it:
    """file_path = "./training_data/grug.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()"""
    
    # Load the data to train on from huggingface:
    dataset = load_dataset("stas/openwebtext-10k")

    # Access the text data (might differ depending on the dataset format)
    text_data = dataset["train"]["text"]  # Assuming the text data is in the "text" column of the training split
    text_data = "\n".join(text_data) # Flatten text to a single string for us to split up later.

    tokenizer, gpt_config = setup_tokenizer(gpt_config, continue_training_from)
    model, optimizer = load_or_initialize_model(gpt_config, settings, continue_training_from, device, compile_model)

    print("Setting up dataloaders...")
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_loader = CreateDataloader(
        gpt_config["tokenizer"],
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = CreateDataloader(
        gpt_config["tokenizer"],
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    print("Training model...")
    startDateTime = time.time()
    total_steps = len(train_loader) * settings["num_epochs"]
    warmup_steps = int(0.2 * total_steps)
    eval_freq = 5
    eval_iter = 1
    save_every = 500

    if warmup_steps == 0:
        warmup_steps = 1
    
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device, n_epochs=settings["num_epochs"],
        eval_freq=eval_freq, eval_iter=eval_iter, start_context="Every effort moves you",
        tokenizer=tokenizer, warmup_steps=warmup_steps, 
        initial_lr=5e-5, min_lr=1e-5, save_every=save_every
    )
    
    endDateTime = time.time()
    timeTaken = endDateTime - startDateTime
    print(f"Time taken: {timeTaken}")
    return train_losses, val_losses, tokens_seen, model

if __name__ == "__main__":

    BAMBOO_CONFIG_365M = {
        "tokenizer": "o200k_base", # Tokenizer to use, default GPT-4o tokenizer
        "context_length": 1536,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 8,         # Number of layers
        "drop_rate": 0.0,       # Dropout rate, disabled since it's no longer recommended for LLMs
        "qkv_bias": False,       # Query-key-value bias,
        "window_size": 1024      # Window size for sliding window attention
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 6,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    model_name = "bamboo-1-365M-grug"
    file_path_folder = f"./output/{model_name}"

    train_losses, val_losses, tokens_seen, model = main(BAMBOO_CONFIG_365M, OTHER_SETTINGS, "", False)
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig(f"{file_path_folder}/loss.jpg")

    # Create folder if it does not exist:
    if not os.path.exists(file_path_folder):
        os.makedirs(file_path_folder)

    # Save the model:
    torch.save(model.state_dict(), f"{file_path_folder}/model.pth")

    # Save the config as a json file:
    with open(f"{file_path_folder}/config.json", "w") as f:
        json.dump(BAMBOO_CONFIG_365M, f)