# Import necessary libraries
import datetime
import math
import time
import numpy as np  # Add this line to import numpy
import matplotlib
matplotlib.use('Agg')  # Add this line to use matplotlib without a display
import matplotlib.pyplot as plt
#from galore_torch import GaLoreAdamW
import os
import torch
import urllib.request
import tiktoken
# Our classes:
from architecture.multiheadattention import MultiHeadAttention
from architecture.transformerblock import LayerNorm, GELU, FeedForward, TransformerBlock
from architecture.gpt import GPTModel
from architecture.dataset import GPTDataset, create_dataloader_v1
import torch_optimizer as optim
import SM3

from torch.cuda.amp import autocast, GradScaler

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    with autocast():
        logits = model(input_batch)
        logits = logits.flatten(0, 1).float()
        target_batch = target_batch.flatten().long().to(device)
        loss = torch.nn.functional.cross_entropy(logits, target_batch)

    return loss

def calc_loss_loader(data_loader, model, device, scaler, optimizer, num_batches=None):
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

            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter, scaler):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, scaler, optimizer=None, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, scaler, optimizer=None, num_batches=eval_iter)
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

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, learning_rates):
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
    # Create a third y-axis for learning rates (secondary y-axis)
    ax3 = ax1.twinx()  # Share the x-axis
    ax3.plot(epochs_seen, learning_rates, color="green", label="Learning rate", linestyle=":", marker='^')
    ax3.set_ylabel("Learning Rate")
    ax3.legend(loc="upper left")
    fig.tight_layout()

def train_model_simple(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    scaler = GradScaler()
    
    for epoch in range(n_epochs):
        model.train()  # Set model to training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            scaler.step(optimizer)
            scaler.update()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def train_model(model, train_loader, val_loader, optimizer, device, n_epochs, eval_freq, eval_iter, start_context, tokenizer, warmup_steps, initial_lr=3e-05, min_lr=1e-6, save_every=5):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * n_epochs
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    scaler = GradScaler()
    
    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment
            else:
                progress = ((global_step - warmup_steps) / (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            with autocast():
                loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter, scaler)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                track_lrs.append(lr)

                print(f"Ep {epoch+1} (Iter {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

        if (epoch + 1) % save_every == 0:
            output_name = f"./output/model_and_optimizer_{epoch+1}.pth"
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, output_name)

    return train_losses, val_losses, track_tokens_seen, track_lrs

def main(gpt_config, settings, continue_training_from="", compile_model=True):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using the GPU! 🔥")
    else:
        print("Using the CPU! ❄️")

    file_path = "./training_data/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    
    if continue_training_from != "":
        print("Continuing training from a previous checkpoint...")
        checkpoint = torch.load(continue_training_from)
        model = GPTModel(GPT_CONFIG_124M)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("Initializing model...")
        model = GPTModel(gpt_config)
    
    #model.half()
    model.to(device)
    model.pos_emb.to(device)
    
    if compile_model:
        print("Compiling model...")
        model = torch.compile(model)
        if os.name == "nt":
            print("Windows OS detected. Setting trinitron fallback to True.")
            torch._dynamo.config.suppress_errors = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    
    print("Setting up dataloaders...")
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    print("Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    print("Training model...")
    startDateTime = time.time()
    total_steps = len(train_loader) * settings["num_epochs"]
    warmup_steps = int(0.2 * total_steps)
    eval_freq = 100
    eval_iter = 1
    save_every = 99999
    train_losses, val_losses, tokens_seen, learning_rates = train_model(
        model, train_loader, val_loader, optimizer, device,
        n_epochs=settings["num_epochs"], eval_freq=eval_freq, eval_iter=eval_iter,
        start_context="Every effort moves you", tokenizer=tokenizer, warmup_steps=warmup_steps, save_every=save_every
    )
    
    endDateTime = time.time()
    timeTaken = endDateTime - startDateTime
    print(f"Time taken: {timeTaken}")
    return train_losses, val_losses, tokens_seen, model, learning_rates

if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 2048,
        "emb_dim": 768,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "window_size": 768  # for example, a window size of 256 tokens
    }
    GPT_CONFIG_350M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "window_size": 256
    }
    GPT_CONFIG_760M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 6,
        "batch_size": 1,
        "weight_decay": 0.1
    }

    train_losses, val_losses, tokens_seen, model, learning_rates = main(GPT_CONFIG_124M, OTHER_SETTINGS, "", False)
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, learning_rates)
    plt.savefig("./output/loss.jpg")
    torch.save(model.state_dict(), "./output/model.pth")