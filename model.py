import os 
import math

# Third-party
import sentencepiece as spm
import torch
import torch.nn as nn 
from torch.nn import functional as F

# -------- CONSTANTS -------- # 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
VOCAB_SIZE = 50257 # 51000
LR = 6e-4 # 3e-4
DROPOUT = 0.2
HEADS = 8
NX = 8
LR = 3e-4 # 6e-4
BATCH_SIZE = 14 # 64
CTX = 200 # 256
EMBED_DIM = 584
# --------------------------- # 

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

class Model(nn.Module):

    def __init__(self, vocab_size=VOCAB_SIZE):
        super().__init__()
        # print(vocab_size)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding_table = PositionalEncoding(EMBED_DIM, DROPOUT, CTX).to(device)
        self.encoder = nn.Sequential(*[Encoder() for _ in range(NX)])
        self.decoder = nn.Sequential(*[Decoder() for _ in range(NX)])
        self.ln_final = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x, targets=None):
        
        B, T = x.shape

        # add embedding to tokens
        tok_enb = self.token_embedding_table(x) # B, T, C
        pos_enb = self.position_embedding_table(torch.transpose(tok_enb, 0, 1).to(device)) # T, B, C
        x = torch.transpose(pos_enb, 1, 0).to(device) # B, T, C

        # Feed into encoder
        enc_out = self.encoder(x) # -> B, T, C

        # Feed into decoder with cross-attention
        x = self.decoder((x, enc_out)) # does this tuple thing work? 
        x = self.ln_final(x)

        # finally plug into language model head
        logits = self.lm_head(x) # B, T, vocab_size 
        # print(logits)

        if targets is None: loss = None
        else:
            # must convert tensor size for logits/targets for each token in vocab to have 
            # activation value between 0 and 1
            batch_size, CTX, vocab_size = logits.shape
            logits = logits.view(batch_size * CTX, vocab_size)
            targets = targets.view(batch_size*CTX)
            loss = F.cross_entropy(logits, targets)
        
        return (logits, loss)
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens):
        
        # x is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop x to last CTX tokens to prevent too much
            x_crop = x[:,-CTX:]
            # get the predictions, ignore loss
            logits, _ = self(x_crop)
            # focus only on the last element in time dimension
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1).to(device) # (B, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, x_next), dim=1) # (B, T+1)
        return x
    
class Encoder(nn.Module):
    """
    Encoder block feature multiheaded attention and feedforward
    """
    def __init__(self):
        super().__init__()
        divided_head_size = EMBED_DIM // HEADS
        self.sa = MultiHeadAttention(divided_head_size)
        self.ffwd = FeedForward(EMBED_DIM)
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
 
        x = self.sa(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x

class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """
    def __init__(self, divided_head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(divided_head_size) for _ in range(HEADS)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        out = torch.concat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Head(nn.Module):
    """
    One head of masked self-attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        
        B, T, C = x.shape
        self.key(x)
  
        key = self.key(x).to(device)        # (B,T,C)
        query = self.query(x).to(device)    # (B,T,C)

        # match query against every key
        weights = query @ key.transpose(-2, -1).to(device) * (C ** -0.5) # (B,T,C) @ (B,C,T) -> (B,T,T) 
        
        # take softmax to determine each token's importance relative to any abritrary token
        weights = F.softmax(weights, dim=-1).to(device) # (B,T,T)
        # print(weights.shape)
        weights = self.dropout(weights)

        # weighted aggregation of the values
        v = self.value(x).to(device) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        # print(out.shape)
        return out

class Decoder(nn.Module):
    """
    Decoder block featuring masked multiheaded attention, cross-attention, and feedforward 
    """
    def __init__(self):
        # EMBED_DIM: embedding dimension
        # n_head: number of heads to use
        super().__init__()
        divided_head_size = EMBED_DIM // HEADS
        self.sa = MaskedMultiHeadAttention(divided_head_size)
        self.ca = MultiHeadCrossAttention(divided_head_size)
        self.ffwd = FeedForward(EMBED_DIM)
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ln3 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x: tuple):
        
        x, enc_out = x
        
        # Masked self-attention
        x = self.sa(self.ln1(x)) + x

        # Cross-attention
        x = self.ca(x, enc_out)
        
        # Computation with residual connection
        x = self.ffwd(self.ln2(x)) + x
        # print(x.shape)
        return x
    
class MaskedMultiHeadAttention(nn.Module):
    """
    Multiple heads of masked self-attention in parallel
    """
    def __init__(self, divided_head_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedHead(divided_head_size) for _ in range(HEADS)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        out = torch.concat([h(x) for h in self.heads], dim=-1)
        # print(out.shape)
        # Linear transformation of outcome 
        out = self.dropout(self.proj(out))
        return out
    
class MaskedHead(nn.Module):
    """
    One head of masked self-attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CTX, CTX).to(device)).to(device))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        
        B, T, C = x.shape
        # print('Head')
        # print(x.shape)

        self.key(x)
  
        key = self.key(x).to(device)        # (B,T,C)
        query = self.query(x).to(device)    # (B,T,C)
        # print(key.shape, query.shape)

        # compute attention scores (affinities)
        # note scores divided by square root of channels to de-sharpen values for softmax later

        # match query against every key
        weights = query @ key.transpose(-2, -1).to(device) * (C ** -0.5) # (B,T,C) @ (B,C,T) -> (B,T,T) 
        # print(weights.shape)
        # optional mask to ignore future tokens
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')).to(device) # (B,T,T)

        # take softmax to determine each token's importance relative to any abritrary token
        weights = F.softmax(weights, dim=-1).to(device) # (B,T,T)
        # print(weights.shape)
        weights = self.dropout(weights)

        # weighted aggregation of the values
        v = self.value(x).to(device) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        # print(out.shape)
        return out
    
class MultiHeadCrossAttention(nn.Module):
    """
    Multiple heads of cross-attention in parallel
    """
    def __init__(self, divided_head_size):
        super().__init__()
        self.heads = nn.ModuleList([CaHead(divided_head_size) for _ in range(HEADS)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        out = torch.concat([h.forward(x, enc_out) for h in self.heads], dim=-1)
        # print(out.shape)
        # Linear transformation of outcome 
        out = self.dropout(self.proj(out))
        return out
    
class CaHead(nn.Module):
    """
    One head of masked self-attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CTX, CTX).to(device)).to(device))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        
        B, T, C = x.shape
        # print('Head')
        # print(x.shape)

        self.key(x)
  
        key = self.key(x).to(device)        # (B,T,C)
        query = self.query(x).to(device)    # (B,T,C)
        # print(key.shape, query.shape)

        # compute attention scores (affinities)
        # note scores divided by square root of channels to de-sharpen values for softmax later

        # match query against every key
        weights = query @ key.transpose(-2, -1).to(device) * (C ** -0.5) # (B,T,C) @ (B,C,T) -> (B,T,T) 
        # print(weights.shape)
        # optional mask to ignore future tokens
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')).to(device) # (B,T,T)

        # take softmax to determine each token's importance relative to any abritrary token
        weights = F.softmax(weights, dim=-1).to(device) # (B,T,T)
        # print(weights.shape)
        weights = self.dropout(weights)

        # weighted aggregation of the values
        v = self.value(x).to(device) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        # print(out.shape)
        return out
    
class FeedForward(nn.Module):
    """
    Linear layer followed by a non-linearity
    """
    def __init__(self, EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.ReLU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)
    
