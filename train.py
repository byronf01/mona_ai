import os 
import sys
from model import Model

# Third-party
import sentencepiece as spm
import torch
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken
import numpy as np

# -------- CONSTANTS -------- # 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
enc = tiktoken.get_encoding("gpt2")
MAX_ITERS = 500000
EVAL_INTERVAL = 5000
SAVE_INTERVAL = 2000
EVAL_ITERS = 150
DROPOUT = 0.2
HEADS = 8
NX = 8
LR = 3e-4 # 6e-4
BATCH_SIZE = 8 # 64
CTX = 200 # 256
EMBED_DIM = 584
train_data = np.memmap(os.path.join('./data', 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join('./data', 'val.bin'), dtype=np.uint16, mode='r')
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
PADDING = 50257
# torch.manual_seed(1337)
# --------------------------- # 

def get_batch(split):
    """
    Generates batch data of BATCH_SIZE of inputs x which is of CTX and targets y
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - CTX, (BATCH_SIZE,), device=device)
    x = torch.stack([torch.from_numpy((data[i:i+CTX]).astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy((data[i+1:i+1+CTX]).astype(np.int64)) for i in ix]).to(device)
    return x, y

def get_unsupervised_batch(split):
    """
    Generates batch data with 0-25 units of padding 
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - CTX, (BATCH_SIZE,), device=device)
    seed = np.random.randint(0, 26)
    x = torch.stack([torch.tensor([PADDING for _ in range(seed)] + list(data[i:i+CTX-seed]), dtype=torch.int64) for i in ix]).to(device)
    y = torch.stack([torch.tensor([PADDING for _ in range(seed)] + list(data[i+1:i+1+CTX-seed]), dtype=torch.int64) for i in ix]).to(device)
    return x, y

def get_mask(idx):
    return torch.tensor([[1 if token != PADDING else 0 for token in tensor] for tensor in idx])

def calculate_loss(logits, targets):

    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)
    loss = F.cross_entropy(logits, targets)
    return loss

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS, device=device)
        for k in range(EVAL_ITERS):
            source, targets = get_unsupervised_batch(split)
            src_mask = get_mask(source)
            pe_mask = get_mask(targets)
            logits = model(source, src_mask, targets, pe_mask)
            loss = calculate_loss(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    

    start = "What's the strangest thing you've ever done"
    start_ids = encode(start)
    while len(start_ids) < CTX: start_ids = [PADDING] + start_ids
    x = torch.tensor([start_ids for _ in range(1)], dtype=torch.long, device=device)
    start_mask = get_mask(x)

    # Setting up the model
    if 'load' not in sys.argv:
        m = Model()
        m.to(device)
        checkpoint = 0
    else:
        iter = sys.argv[-1]
        m = torch.load(f'models/epoch{iter}.pth')
        checkpoint = int(iter)
    """
    print('Before: ')
    m.eval()
    print(decode( m.generate(x, start_mask)) )
    m.train()
    """
    # create torch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for iter in range(checkpoint, MAX_ITERS):

        source, targets = get_unsupervised_batch('train')
        src_mask = get_mask(source)
        pe_mask = get_mask(targets)
        logits = m(source, src_mask, targets, pe_mask)
        optimizer.zero_grad(set_to_none=True)

        loss = calculate_loss(logits, targets)

        loss.backward()
        optimizer.step()

        # periodically evaluate loss on training and validation sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # periodically save the model
        if iter % SAVE_INTERVAL == 0:
            torch.save(m, f'models/epoch{iter}.pth')
        
    print('After: ')
    print(decode(m.generate(x, max_new_tokens=3000)[0].tolist() ))
