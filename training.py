# Import necessary libraries
import datetime
import json
import math
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
import os
import torch
import tiktoken
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# Our classes:
from architecture.gpt import GPTModel
from architecture.dataset import BambooDataset, CreateDataloader

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    with torch.cuda.amp.autocast(enabled=model.dtype in [torch.float16, torch.bfloat16]):
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, dtype, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.cuda.amp.autocast(enabled=dtype != torch.float32, dtype=dtype):
                logits = model(input_batch)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
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
    ax1.plot(epochs_seen, train_losses, label="Training loss", marker='o')
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss", marker='x')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()

def train_model(model, train_loader, val_loader, optimizer, device, dtype,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6, save_every=500, world_size=1, rank=0):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * n_epochs
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    # Use GradScaler only for float16
    use_scaler = False #dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()

        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

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
            track_lrs.append(lr)

            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.cuda.amp.autocast(enabled=dtype != torch.float32, dtype=dtype):
                logits = model(input_batch)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

            if use_scaler:
                scaler.scale(loss).backward()
                if global_step > warmup_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            tokens_seen += input_batch.numel()

            batch_time = time.time() - start_time         

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, dtype, num_batches=eval_iter)
                    val_loss = calc_loss_loader(val_loader, model, device, dtype, num_batches=eval_iter)
                model.train()

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                steps_per_second = 1.0 / batch_time
                if rank == 0:
                    print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}, "
                          f"Steps/s: {steps_per_second:.2f}"
                    )

            if global_step > 0 and global_step % save_every == 0 and rank == 0:
                out_folder = "./output/"
                torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), f"{out_folder}/model.pth")
                torch.save(optimizer.state_dict(), f"{out_folder}/optimizer.pth")

                with open(f"{out_folder}/config.json", "w") as f:
                    json.dump(model.module.config if isinstance(model, DDP) else model.config, f)

                print(f"Saved model & optimizer state dict @ step: {global_step}")

        if rank == 0:
            generate_and_print_sample(model.module if isinstance(model, DDP) else model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen, track_lrs

def setup_tokenizer(gpt_config, checkpoint_path):
    print("Loading tokenizer...")
    if checkpoint_path:
        config_file = f"{checkpoint_path}/config.json"
        with open(config_file, "r") as f:
            checkpoint_config = json.load(f)
        tokenizer = tiktoken.get_encoding(checkpoint_config["tokenizer"])
        gpt_config = checkpoint_config
    else:
        tokenizer = tiktoken.get_encoding(gpt_config["tokenizer"])
    
    gpt_config["vocab_size"] = tokenizer.n_vocab
    return tokenizer, gpt_config

def load_or_initialize_model(gpt_config, settings, continue_training_from, device, compile_model, rank, world_size):
    if continue_training_from:
        print("Continuing training from a previous checkpoint...")
        model = GPTModel(gpt_config)
        checkpoint = torch.load(f"{continue_training_from}/model.pth", map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("Initializing new model...")
        model = GPTModel(gpt_config)

    # Determine the appropriate dtype
    if device.type == "cuda":
        if torch.cuda.get_device_capability()[0] >= 8:
            print("Converting model to bfloat16...")
            dtype = torch.bfloat16
        elif torch.cuda.get_device_capability()[0] >= 6:
            print("Converting model to float16...")
            dtype = torch.float16
        else:
            print("Using default float32...")
            dtype = torch.float32
    else:
        dtype = torch.float32

    model = model.to(device=device, dtype=dtype)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if compile_model and not continue_training_from:
        model = compile_model_based_on_os(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    
    if continue_training_from:
        optimizer.load_state_dict(torch.load(f"{continue_training_from}/optimizer.pth", map_location=device))

    return model, optimizer, dtype

def compile_model_based_on_os(model):
    print("Compiling model...")
    if os.name == "nt":
        print("Windows OS detected. Compiling as torch.compile() + Setting trinitron fallback to True.")
        model = torch.compile(model)
        torch._dynamo.config.suppress_errors = True
    else:
        model = thunder.jit(model)
    return model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_available_gpus():
    return torch.cuda.device_count()

def main(rank, world_size, gpt_config, settings, continue_training_from="", compile_model=True):
    if world_size > 1:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(123)
    if rank == 0:
        print(f"Using {'GPU! 🔥' if device.type == 'cuda' else 'CPU! ❄️'}")

    dataset = load_dataset(settings["dataset_name"])
    text_data = dataset["train"]["text"]
    text_data = "\n".join(text_data)

    tokenizer, gpt_config = setup_tokenizer(gpt_config, continue_training_from)
    model, optimizer, dtype = load_or_initialize_model(gpt_config, settings, continue_training_from, device, compile_model, rank, world_size)

    if rank == 0:
        print("Setting up dataloaders...")
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    if world_size > 1:
        train_sampler = DistributedSampler(BambooDataset(text_data[:split_idx], tokenizer, gpt_config["context_length"], gpt_config["context_length"]))
        val_sampler = DistributedSampler(BambooDataset(text_data[split_idx:], tokenizer, gpt_config["context_length"], gpt_config["context_length"]))
    else:
        train_sampler = None
        val_sampler = None

    train_loader = CreateDataloader(
        gpt_config["tokenizer"],
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = CreateDataloader(
        gpt_config["tokenizer"],
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    if rank == 0:
        print("Training model...")
    startDateTime = time.time()
    total_steps = len(train_loader) * settings["num_epochs"]
    warmup_steps = max(1, int(0.2 * total_steps))
    eval_freq = 5
    eval_iter = 1
    save_every = 500

    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device, dtype,
        n_epochs=settings["num_epochs"],
        eval_freq=eval_freq, eval_iter=eval_iter, start_context="Every effort moves you",
        tokenizer=tokenizer, warmup_steps=warmup_steps, 
        initial_lr=settings["learning_rate"], min_lr=1e-5, save_every=save_every,
        world_size=world_size, rank=rank
    )
    
    endDateTime = time.time()
    timeTaken = endDateTime - startDateTime
    if rank == 0:
        print(f"Time taken: {timeTaken}")

    if world_size > 1:
        cleanup()

    return train_losses, val_losses, tokens_seen, model

if __name__ == "__main__":
    BAMBOO_CONFIG_365M = {
        "tokenizer": "o200k_base",
        "context_length": 1536,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 8,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "window_size": 1024
    }

    OTHER_SETTINGS = {
        "learning_rate": 1e-5,
        "num_epochs": 12,
        "batch_size": 1,
        "weight_decay": 0.1,
        #"dataset_name": "stas/openwebtext-10k"
        "dataset_name": "./training_data/the-verdict.txt"
    }

    model_name = "bamboo-365M-grug"
    file_path_folder = f"./output/{model_name}"

    if len(sys.argv) == 5:
        model_name = sys.argv[1]
        OTHER_SETTINGS["num_epochs"] = int(sys.argv[2])
        OTHER_SETTINGS["batch_size"] = int(sys.argv[3])
        OTHER_SETTINGS["dataset_name"] = sys.argv[4]
        print(OTHER_SETTINGS)
    else:
        print("Usage: model_name, num_epochs, batch_size, dataset_name")

    world_size = get_available_gpus()
    if world_size == 0:
        print("No GPUs available. Running on CPU.")
        train_losses, val_losses, tokens_seen, model = main(0, 1, BAMBOO_CONFIG_365M, OTHER_SETTINGS, "", False)
    elif world_size == 1:
        print("Running on 1 GPU")
        train_losses, val_losses, tokens_seen, model = main(0, 1, BAMBOO_CONFIG_365M, OTHER_SETTINGS, "", False)
    else:
        print(f"Running on {world_size} GPUs")
        OTHER_SETTINGS["learning_rate"] *= world_size
        print(f"Adjusted learning rate for {world_size} GPUs: {OTHER_SETTINGS['learning_rate']}")
        mp.spawn(main, args=(world_size, BAMBOO_CONFIG_365M, OTHER_SETTINGS, "", False), nprocs=world_size)

    if world_size <= 1 or (world_size > 1 and torch.distributed.get_rank() == 0):
        # Create folder if it does not exist:
        if not os.path.exists(file_path_folder):
            os.makedirs(file_path_folder)

        # Plot and save losses
        epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig(f"{file_path_folder}/loss.jpg")

        # Save the model:
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), f"{file_path_folder}/model.pth")
        else:
            torch.save(model.state_dict(), f"{file_path_folder}/model.pth")

        # Save the config as a json file:
        with open(f"{file_path_folder}/config.json", "w") as f:
            json.dump(BAMBOO_CONFIG_365M, f)

        print(f"Model, losses, and config saved in {file_path_folder}")