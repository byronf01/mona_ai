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
HEADS = 8
N_LAYER = 8
LR = 3e-4
MAX_ITERS = 500000
EVAL_INTERVAL = 2000
SAVE_INTERVAL = 10000
EVAL_ITERS = 200
DROPOUT = 0.2
BATCH_SIZE = 14 # 64
CTX = 200 # 256
EMBED_DIM = 584
train_data = np.memmap(os.path.join('./data', 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join('./data', 'val.bin'), dtype=np.uint16, mode='r')
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
torch.manual_seed(1337)
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

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS, device=device)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            logits, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    
    """

    with open('test.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # set up encoding
    encoding = enc.encode(text) 
    vocab_size = len(set(encoding))

    data = torch.tensor(encoding, dtype=torch.long, device=device)

    n = int(0.9*len(data))
    train_data = data[:n]
    validation_data = data[n:]

    """
    
    start = 'hii there'
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # Setting up the model
    if 'load' not in sys.argv:
        m = Model()
        m.to(device)
    else:
        iter = sys.argv[-1]
        m = torch.load(f'models/epoch{iter}.pth')

    print('Before: ')
    m.eval()
    # print(decode( m.generate(x, max_new_tokens=1000)[0].tolist() ) )
    m.train()
    
    # create torch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for iter in range(MAX_ITERS):

        xb, yb = get_batch('train')
        # print(xb.shape, yb.shape)
        logits, loss = m(xb, yb)
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
