import os 
import sys
import time
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
EVAL_INTERVAL = 100
SAVE_INTERVAL = 1000
EVAL_ITERS = 50
DROPOUT = 0.2
HEADS = 8
NX = 8
LR = 2e-5 # 6e-4
BATCH_SIZE = 14
CTX = 200
EMBED_DIM = 512
train_data = np.memmap(os.path.join('./data', 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join('./data', 'val.bin'), dtype=np.uint16, mode='r')
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
PADDING = 50257
START_TOKEN = 50258
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

def get_unsupervised_batch(split):
    """
    Generates batch data with 0-25 units of padding each. Structure of data is:
    source = [padding + Data]
    teacher = [padding + Start token + Data immediately following indice from input data]
    targets = [padding + Teacher data shifted right by 1]
    """
    max_padding = 26
    data = train_data if split == 'train' else val_data

    # account for encoder input & decoder input both being max length of CTX
    ix = torch.randint(len(data) - 2 * CTX, (BATCH_SIZE,), device=device)
    p1, p2 = (np.random.randint(0, max_padding), np.random.randint(1, max_padding + 1))
    source = torch.stack([torch.tensor([PADDING for _ in range(p1)] + list(data[i:i + CTX - p1]), dtype=torch.int64) for i in ix]).to(device)
    teacher = torch.stack([torch.tensor([PADDING for _ in range(p2-1)] + [START_TOKEN] + list(data[i + CTX - p1:i - p1 - p2 + 2 * CTX]), dtype=torch.int64) for i in ix]).to(device)
    targets = torch.stack([torch.tensor([PADDING for _ in range(p2-1)] + list(data[i + CTX - p1:i - p1 - p2 + 2 * CTX + 1]), dtype=torch.int64) for i in ix]).to(device)
    return source, teacher, targets

def get_mask(idx):
    return torch.tensor([[1 if token != PADDING else 0 for token in tensor] for tensor in idx]).to(device)

def calculate_loss(logits, targets, padding_mask):

    B, T, C = logits.shape
    """
    # Apply mask to logits and targets based on padding tokens in targets
    logits = logits * padding_mask.unsqueeze(2)
    targets = targets * padding_mask
    """
    # Resizing of logits and targets
    logits = logits.view(B * T, C).to(device)
    targets = targets.view(B * T).to(device)
    ignore_mask = padding_mask.view(B * T).to(device)
    # print(logits.shape, targets.shape, weights.shape)
    # print(weights)
    # Weights?

    loss1 = F.cross_entropy(logits, targets)

    loss2 = torch.tensor(0.0, device=logits.device)
    valid_batches = 0
    for i in range(len(logits)):
        if ignore_mask[i] == 1:
            loss = F.cross_entropy(logits[i], targets[i])
            loss2 += loss
            valid_batches += 1
    loss2 = loss2 / valid_batches if valid_batches > 0 else 0.0

    # print(loss1, loss2)
    return loss2
    

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS, device=device)
        for k in range(EVAL_ITERS):
            source, teacher,targets = get_unsupervised_batch(split)
            src_mask = get_mask(source)
            teacher_mask = get_mask(teacher)
            target_mask = get_mask(targets)
            logits = model(source, src_mask, targets, teacher_mask)
            loss = calculate_loss(logits, targets, target_mask)
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
    out = m.generate(x, start_mask)
    out = [0 if token == PADDING else token for token in out]
    print(decode( out ))
    m.train()
    """
    start = time.time()
    # create torch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=LR)
    for iter in range(checkpoint, MAX_ITERS):

        source, teacher, targets = get_unsupervised_batch('train')
        src_mask = get_mask(source)
        teacher_mask = get_mask(teacher)
        target_mask = get_mask(targets) # do we need this?
        logits = m(source, src_mask, teacher, teacher_mask)
        optimizer.zero_grad(set_to_none=True)

        loss = calculate_loss(logits, targets, target_mask)

        loss.backward()
        optimizer.step()

        # periodically evaluate loss on training and validation sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, {(time.time() - start):.5f} time elapsed")

        # periodically save the model
        if iter % SAVE_INTERVAL == 0:
            torch.save(m, f'models/epoch{iter}.pth')
        
    print('After: ')
    print(decode(m.generate(x, max_new_tokens=3000)[0].tolist() ))
