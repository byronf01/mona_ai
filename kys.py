import os 
import sys
import time
import math

# Third-party
import sentencepiece as spm
import torch
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken
import numpy as np

# -------- HYPERPARMETERS --------- # 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = 50257 + 1 + 1 # 51000
MAX_ITERS = 500000
EVAL_INTERVAL = 500
SAVE_INTERVAL = 200
EVAL_ITERS = 50
DROPOUT = 0.1
HEADS = 8
NX = 8
LR = 1e-5 # 6e-4
BATCH_SIZE = 14 # 8
CTX = 200 # 52
EMBED_DIM = 568 # 128
train_data = np.memmap(os.path.join('./data', 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join('./data', 'val.bin'), dtype=np.uint16, mode='r')
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
PADDING = 50257
START_TOKEN = 50258
torch.manual_seed(1337)
# --------------------------------- # 

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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
    
    source, teacher = source.unsqueeze(-1).repeat(1, 1, EMBED_DIM), teacher.unsqueeze(-1).repeat(1, 1, EMBED_DIM)
    return (source, teacher, targets)

def get_masks(idx: [torch.tensor]) -> (torch.tensor):
    """
    Returns a tuple of masks for all tensors in argument
    """
    return tuple(torch.tensor([[torch.ones(EMBED_DIM) if token[0] != PADDING else torch.zeros(EMBED_DIM) for token in tensor] for tensor in i]).to(device) for i in idx)

def calculate_loss(logits, targets, padding_mask):

    B, T, C = logits.shape
    
    # Resizing of logits and targets
    logits = logits.view(B * T, C).to(device)
    targets = targets.view(B * T).to(device)
    ignore_mask = padding_mask.view(B * T).to(device)
    # print(logits.shape, targets.shape, weights.shape)
    # print(weights)
    # Weights?

    # loss1 = F.cross_entropy(logits, targets)

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

def generate(model, x, src_mask):
    """
    Generates output from the model. Stops generating when end of text token generates.
    x: 1 x N tensor
    """
    
    output = []
    predict = torch.tensor([[PADDING for _ in range(CTX)] for _ in range(1)]).to(device)
    end = False
    failsafe = CTX + int(0.5 * CTX)

    while True:
        
        # Make mask for decoder outputs 
        pred_mask = torch.tensor([[0 if token == PADDING else 1 for token in tensor] for tensor in predict]).to(device)

        # One step in model
        logits = model(x, predict, src_mask, pred_mask)
        print(logits.shape)

        probs = F.softmax(logits[:, -1, :], dim=-1)

        x_next = torch.multinomial(probs, num_samples=1).to(device)

        if x_next[0][0] == enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
            end = True

        # Move newly generated token to predicted sequence
        predict = torch.cat((predict, x_next), dim=1)
        predict = predict[:,-CTX:]
        output += x_next.tolist()[0]

        if end: break

        if failsafe > 0: failsafe -= 1
        else: break

    return output

if __name__ == '__main__':

    
    start = "What's the strangest thing you've ever done"
    start_ids = encode(start)
    while len(start_ids) < CTX: start_ids = [PADDING] + start_ids
    x = torch.tensor([start_ids for _ in range(1)], dtype=torch.long, device=device).unsqueeze(-1).repeat(1, 1, EMBED_DIM)
    print(x.shape)
    start_mask = get_masks([x])[0]

    # Setting up the model
    if 'load' not in sys.argv:
        m = nn.Transformer(d_model=EMBED_DIM, nhead=HEADS, num_encoder_layers=NX, num_decoder_layers=NX,
                                       dim_feedforward=2048, dropout=DROPOUT, activation=nn.ReLU(), batch_first=True, 
                                       norm_first=True, device=None, dtype=torch.float64)
        m.to(device)
        checkpoint = 0
    else:
        iter = sys.argv[-1]
        m = torch.load(f'models/epoch{iter}.pth')
        checkpoint = int(iter)
    
    print('Before: ')
    m.eval()
    out = generate(m, x, start_mask)
    out = [0 if token == PADDING else token for token in out]
    print(decode( out ))
    m.train()
    
    start = time.time()
    # create torch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=LR)
    for iter in range(checkpoint, MAX_ITERS):

        source, teacher, targets = get_unsupervised_batch('train')
        src_mask, teacher_mask, target_mask = get_masks([source, teacher, targets])
        logits = m(source, teacher, src_mask, teacher_mask)
        print(type(logits))
        print(logits.shape)
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

