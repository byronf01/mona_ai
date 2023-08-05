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
    Generates batch data with 5 units of padding at the end
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - CTX, (BATCH_SIZE,), device=device)
    x = torch.stack([torch.tensor(list(data[i:i+CTX-5]) + [PADDING for _ in range(5)], dtype=torch.int64) for i in ix]).to(device)
    y = torch.stack([torch.tensor(list(data[i+1:i+1+CTX-5]) + [PADDING for _ in range(5)], dtype=torch.int64) for i in ix]).to(device)
    return x, y

def get_src_mask(idx):
    return torch.tensor([[1 if token != PADDING else 0 for token in tensor] for tensor in idx])

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS, device=device)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            src_mask = get_src_mask(x)
            logits, loss = model(x, src_mask, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    
    start = "What's the strangest thing you've ever done"
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    start_mask = get_src_mask(x)

    # Setting up the model
    if 'load' not in sys.argv:
        m = Model()
        m.to(device)
        checkpoint = 0
    else:
        iter = sys.argv[-1]
        m = torch.load(f'models/epoch{iter}.pth')
        checkpoint = int(iter)

    print('Before: ')
    m.eval()
    print(decode( m.generate_fixed(x, start_mask, max_new_tokens=100)[0].tolist() ) )
    m.train()
    
    # create torch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for iter in range(checkpoint, MAX_ITERS):

        xb, yb = get_unsupervised_batch('train')
        src_mask = get_src_mask(xb)
        logits, loss = m(xb, src_mask, yb)
        optimizer.zero_grad(set_to_none=True)
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
