import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class BambooDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the input into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    def save(self, path):
        torch.save(self, path)

    def load(path):
        return torch.load(path)


def CreateDataloader(tokenizer, txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0, sampler=None, pin_memory=False):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding(tokenizer)
    # Create dataset
    dataset = BambooDataset(txt, tokenizer, max_length, stride)
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(shuffle if sampler is None else False),
        drop_last=drop_last, 
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=pin_memory
    )
    return dataloader