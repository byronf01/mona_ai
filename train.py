import os 
import sys
from model import Model

# Third-party
import sentencepiece as spm
import torch
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken

# -------- CONSTANTS -------- # 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
HEADS = 6
N_LAYER = 6
LR = 3e-4
MAX_ITERS = 6000
EVAL_INTERVAL = 500
SAVE_INTERVAL = 2000
EVAL_ITERS = 200
DROPOUT = 0.2
BATCH_SIZE = 6
CTX = 8
EMBED_DIM = 32

# --------------------------- # 

def get_batch(split, train_data, val_data):
    """
    Generates batch data of BATCH_SIZE of inputs x which is of CTX and targets y
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - CTX, (BATCH_SIZE,), device=device)
    x = torch.stack([data[i:i+CTX] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+CTX+1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, t, v):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS, device=device)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split, t, v)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    
    with open('test.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # set up encoding
    encoding = enc.encode(text) 
    vocab_size = len(set(encoding))

    data = torch.tensor(encoding, dtype=torch.long, device=device)

    n = int(0.9*len(data))
    train_data = data[:n]
    validation_data = data[n:]

    

    # Setting up the model
    if 'load' not in sys.argv:
        m = Model()
        m.to(device)
    else:
        iter = sys.argv[-1]
        m = torch.load(f'models/epoch{iter}.pth')

    print('Before: ')
    m.eval()
    print(enc.decode(m.generate(x=torch.zeros((1,1),dtype=torch.long,device=device), max_new_tokens=1000)[0].tolist() ))
    m.train()
    
    # create torch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for iter in range(MAX_ITERS):

        xb, yb = get_batch('train', train_data, validation_data)
        # print(xb.shape, yb.shape)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # periodically evaluate loss on training and validation sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(m, train_data, validation_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # periodically save the model
        if iter % SAVE_INTERVAL == 0:
            torch.save(m, f'models/epoch{iter}.pth')
        

    print('After: ')
    print(enc.decode(torch.zeros((1,1),dtype=torch.long,device=device), max_new_tokens=3000)[0].tolist() )
