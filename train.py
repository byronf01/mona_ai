import os 
import sys
import time
import math

# Third-party
import sentencepiece as spm
import torch
import torch.nn as nn 
from torch.nn import functional as F
from copy import deepcopy
import tiktoken
import numpy as np



# -------- HYPERPARMETERS --------- # 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = 51000 # 50257 + 1 + 1 # 51000 used because the model will stop generating the really high numbers after a while anyways?
MAX_ITERS = 100000
EVAL_INTERVAL = 200
SAVE_INTERVAL = 100
EVAL_ITERS = 50
DROPOUT = 0.15 # Yeah
HEADS = 8
NX = 6
LR_MAX = 7e-4 # 6e-4
WARMUP_STEPS = 4000
BATCH_SIZE = 66 # 8
SRC_SEQ_LEN = 128 # 52
TGT_SEQ_LEN = 186 
EMBED_DIM = 512 # 128
FORWARD_DIM = 2048
MAX_PADDING = 5
GRAD_CLIP = 1
train_data = np.memmap(os.path.join('./data', 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join('./data', 'val.bin'), dtype=np.uint16, mode='r')
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
PADDING_TOKEN = 50257
START_TOKEN = 50258
DTYPE = torch.float64
# torch.manual_seed(1337)
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
    Generates batch data with (0, max padding) units of padding each. Structure of data is:
    src = [padding + Data]
    tgt = [padding + Start token + Data immediately following indice from input data]
    exp = [padding + Teacher data shifted right by 1]
    """
    data = train_data if split == 'train' else val_data

    # generate random indicies
    ix = torch.randint(len(data) - (SRC_SEQ_LEN + TGT_SEQ_LEN), (BATCH_SIZE,), device=device)
    p1, p2 = (np.random.randint(0, MAX_PADDING + 1), np.random.randint(1, MAX_PADDING + 1))
    src = torch.stack([torch.tensor([PADDING_TOKEN for _ in range(p1)] + list(data[i:i + SRC_SEQ_LEN - p1]), dtype=torch.int64) for i in ix]).to(device)
    tgt = torch.stack([torch.tensor([PADDING_TOKEN for _ in range(p2-1)] + [START_TOKEN] + list(data[i + SRC_SEQ_LEN - p1:i - p1 - p2 + SRC_SEQ_LEN + TGT_SEQ_LEN]), dtype=torch.int64) for i in ix]).to(device)
    exp = torch.stack([torch.tensor([PADDING_TOKEN for _ in range(p2-1)] + list(data[i + SRC_SEQ_LEN - p1:i - p1 - p2 + SRC_SEQ_LEN + TGT_SEQ_LEN + 1]), dtype=torch.int64) for i in ix]).to(device)
    
    # idk random error might happen one day
    assert src.shape == torch.Size([BATCH_SIZE, SRC_SEQ_LEN])
    assert tgt.shape == torch.Size([BATCH_SIZE, TGT_SEQ_LEN])
    assert exp.shape == torch.Size([BATCH_SIZE, TGT_SEQ_LEN])

    # Get masks before embedding the layers
    src_key_padding_mask, tgt_key_padding_mask = get_input_masks(src, tgt)

    return (src, tgt, exp, src_key_padding_mask, tgt_key_padding_mask)

def get_input_masks(src, tgt) -> (torch.tensor):
    """
    Returns src_key_padding_mask and tgt_key_padding_mask based on unembedded src and tgt input tensors
    """
    src_key_padding_mask = torch.tensor([[True if token == PADDING_TOKEN else False for token in tensor] for tensor in src], device=device)
    tgt_key_padding_mask = torch.tensor([[True if token == PADDING_TOKEN else False for token in tensor] for tensor in tgt], device=device)
    return (src_key_padding_mask, tgt_key_padding_mask)

def calculate_loss(logits, exp):

    T, B, C = logits.shape
    logits = logits.reshape(T * B, C).to(device)
    exp = torch.transpose(exp, 0, 1).to(device)
    exp = exp.reshape(T * B).to(device)

    # Ignore tokens not valid in vocabulary (assums that padding token is set to first number above the vocab size of gpt2)
    weights = torch.tensor([1.0 for _ in range(PADDING_TOKEN)] + [0.0 for _ in range(VOCAB_SIZE - PADDING_TOKEN)], device=device) 
    loss = F.cross_entropy(logits, exp, weight=weights)
    return(loss)

@torch.no_grad()
def estimate_loss(model, tgt_mask):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS, device=device)
        for k in range(EVAL_ITERS):
            src, tgt, exp, src_k_pad_mask, tgt_k_pad_mask = get_unsupervised_batch('train')
            tgt_mask_copy = torch.clone(tgt_mask)
            mem_k_pad_mask = torch.clone(src_k_pad_mask)
            logits = model(src, tgt, tgt_mask=tgt_mask_copy, src_key_padding_mask=src_k_pad_mask, tgt_key_padding_mask=tgt_k_pad_mask, 
                memory_key_padding_mask=mem_k_pad_mask)
            loss = calculate_loss(logits, exp)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def generate(model, prompt: str):
    """
    Generates output from the model. Stops generating when end of text token generates.
    """
    model.eval()
    # Encode the prompt
    prompt = encode(prompt)
    src = torch.tensor([[PADDING_TOKEN for _ in range(SRC_SEQ_LEN - len(prompt) )] + prompt], dtype=torch.long, device=device)
    
    # Starting variables
    output_txt = ''
    end_of_txt = False
    max_chars = TGT_SEQ_LEN * 2 
    src_mask = torch.tensor([[True if token == PADDING_TOKEN else False for token in tensor] for tensor in src], device=device)
    tgt_mask = base.masked_fill(torch.triu(torch.ones(TGT_SEQ_LEN, TGT_SEQ_LEN)) == 0, True).to(device)

    # Tgt tensor starts empty
    tgt = torch.tensor([[PADDING_TOKEN for _ in range(TGT_SEQ_LEN - 1)] + [START_TOKEN] for _ in range(1)]).to(device) 

    while True:
        
        # Make masks 
        tgt_k_pad_mask = torch.tensor([[True if token == PADDING_TOKEN else False for token in tensor] for tensor in tgt], device=device)
        mem_k_pad_mask = torch.clone(src_mask)

        # One step in model
        logits = model(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_k_pad_mask, 
               memory_key_padding_mask=mem_k_pad_mask)

        # Make sense of probabilities
        logits = torch.transpose(logits, 0, 1)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next = torch.multinomial(probs, num_samples=1).to(device)

        # Check specific cases for tokens
        if next[0][0] == enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]:
            end_of_txt = True
        elif next[0][0] >= PADDING_TOKEN: 
            output_txt += '<|?|>'
        else:
            token = next.tolist()[0]
            output_txt += decode(token)

        # Move newly generated token to predicted sequence
        tgt = torch.cat((tgt, next), dim=1)
        tgt = tgt[:,-TGT_SEQ_LEN:]
        tgt.to(device)

        if end_of_txt: break

        if max_chars > 0: 
            max_chars -= 1
        else: 
            break

    model.train()
    return output_txt

class TransformerModel(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, 
                 dropout=0.1, activation=nn.ReLU(), custom_encoder=None, custom_decoder=None, 
                 layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None):
        
        super().__init__()
        self.src_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.tgt_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_encode = PositionalEncoding(EMBED_DIM).to(device)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, 
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, 
                                    activation=activation, custom_encoder=custom_encoder, custom_decoder=custom_decoder,
                                    layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first,
                                    device=device, dtype=dtype)
        self.final_layer = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Arguments:
        src: ```[batch, src seq]```
        tgt: ```[batch, tgt seq]```
        the rest are same as torch.nn.Transformer
        """
        # Pass through first embedding layer
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        # Change dimensions of src and tgt for model
        src = torch.transpose(src, 0, 1).to(device)
        tgt = torch.transpose(tgt, 0, 1).to(device)

        # Add positional encoding
        src = self.pos_encode(src)
        tgt = self.pos_encode(tgt)

        # Forward pass through torch transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                        src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
        
        # Pass output through final linear layer
        output = self.final_layer(output)
        return output

def get_lr(iter):
    """
    learning rate decay scheduler (linear warmup then inverse square)
    """
    if iter < WARMUP_STEPS:
        lr = iter / WARMUP_STEPS * LR_MAX 
    else:
        lr = (EMBED_DIM ** -0.5) * min(iter ** -0.5, iter * (WARMUP_STEPS ** -1.5) )
    return lr


if __name__ == '__main__':

    base = torch.tensor([[False for _ in range(TGT_SEQ_LEN)] for _ in range(TGT_SEQ_LEN)])
    tgt_mask = base.masked_fill(torch.triu(torch.ones(TGT_SEQ_LEN, TGT_SEQ_LEN)) == 0, True).to(device)

    # Setting up the model
    model = TransformerModel(d_model=EMBED_DIM, nhead=HEADS, num_encoder_layers=NX, num_decoder_layers=NX,
                                       dim_feedforward=FORWARD_DIM, dropout=DROPOUT, activation=nn.ReLU(),
                                       norm_first=True, device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX)

    if 'load' not in sys.argv:
        checkpoint = 0
    else:
        iter = sys.argv[-1]

        # Load model and optimizer states from checkpoint
        loaded = torch.load(f'models/epoch{iter}.pth', map_location=device)
        model.load_state_dict(loaded['model_state_dict'])
        optimizer.load_state_dict(loaded['optim_state_dict'])
        checkpoint = int(iter)

        # Free up memory
        del loaded
        if device == 'cuda:0':
            torch.cuda.empty_cache()
        
    # Sample from the model
    response = generate(model, "What is the strangest thing that has ever happened to you or someone you know? ")
    print("Q: What is the strangest thing that has ever happened to you or someone you know?")
    print("A: " + response)
    print(len(response))
    
    start = time.time()
    model.train()
    for iter in range(checkpoint + 1, MAX_ITERS):

        # Set new learning rate
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        src, tgt, exp, src_k_pad_mask, tgt_k_pad_mask = get_unsupervised_batch('train')
        
        # Get other masks
        tgt_mask_clone = torch.clone(tgt_mask)
        mem_k_pad_mask = torch.clone(src_k_pad_mask)
        
        # Get logits from model
        logits = model(src, tgt, tgt_mask=tgt_mask_clone, src_key_padding_mask=src_k_pad_mask, tgt_key_padding_mask=tgt_k_pad_mask, 
               memory_key_padding_mask=mem_k_pad_mask)
        loss = calculate_loss(logits, exp)
        with open('graph_data.txt', 'a') as f:
            f.write(f'Epoch {iter}: {str(loss.item())}\n')

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Free up memory
        optimizer.zero_grad(set_to_none=True)

        # periodically save the model
        if iter % SAVE_INTERVAL == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
            }, f'models/epoch{iter}.pth')

        # periodically evaluate loss on training and validation sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, tgt_mask)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, {(time.time() - start):.5f} time elapsed")

        