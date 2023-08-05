import os 
import math

# Third-party
import sentencepiece as spm
import torch
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken

# -------- CONSTANTS -------- # 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
VOCAB_SIZE = 50257 + 1 # 51000
MASKING_TOKEN = 50257 
LR = 6e-4 # 3e-4
DROPOUT = 0.2
HEADS = 8
NX = 8
LR = 3e-4 # 6e-4
BATCH_SIZE = 14 # 64
CTX = 200 # 256
EMBED_DIM = 584
enc = tiktoken.get_encoding("gpt2")
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

    def forward(self, x, src_mask, targets=None):
        
        B, T = x.shape

        # add embedding to tokens
        tok_enb = self.token_embedding_table(x) # B, T, C
        pos_enb = self.position_embedding_table(torch.transpose(tok_enb, 0, 1).to(device)) # T, B, C
        x = torch.transpose(pos_enb, 1, 0).to(device) # B, T, C

        # Feed logits and padding mask into encoder
        memory, _ = self.encoder((x, src_mask)) # -> B, T, C

        # Feed logits, padding mask and encoder outputs into decoder 
        predict = x # PREDICT SHOULD BE WHAT THE DECODER HAS PREDICTED SO FAR 
        x, _, _ = self.decoder((predict, memory, src_mask))  
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
    def generate_fixed(self, x, src_mask, max_new_tokens):
        
        # x is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop x to last CTX tokens to prevent too much
            x_crop = x[:,-CTX:]
            # get the predictions, ignore loss
            logits, _ = self(x_crop, src_mask)
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
        self.sa = MultiHeadAttention(divided_head_size, mask=False)
        self.ffwd = FeedForward(EMBED_DIM)
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x: (torch.tensor, torch.tensor)):
        
        idx = x[0]
        mask = x[1] # Padding mask

        idx = self.sa((self.ln1(idx), mask)) + idx
        idx = self.ffwd(self.ln2(idx)) + idx
        return (idx, mask)

class Decoder(nn.Module):
    """
    Decoder block featuring masked multiheaded attention, cross-attention, and feedforward 
    """
    def __init__(self):
        # EMBED_DIM: embedding dimension
        # n_head: number of heads to use
        super().__init__()
        divided_head_size = EMBED_DIM // HEADS
        self.sa = MultiHeadAttention(divided_head_size, mask=True)
        self.ca = CrossAttention(divided_head_size, mask=False)
        self.ffwd = FeedForward(EMBED_DIM)
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ln3 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x: tuple):
        
        idx, memory, mask = x
        # Masked self-attention
        idx = self.sa((self.ln1(idx), mask)) + idx

        # Cross-attention
        idx = self.ca(self.ln2(idx), memory) + idx 
        
        # Computation with residual connection
        idx = self.ffwd(self.ln3(idx)) + idx
        
        return (idx, memory, mask)
    
class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """
    def __init__(self, divided_head_size, mask=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(divided_head_size, mask) for _ in range(HEADS)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: (torch.tensor, torch.tensor)):
        
        idx = x[0]
        mask = x[1]
        # print(idx.shape)

        out = torch.concat([h((idx, mask)) for h in self.heads], dim=-1)
        # print(out.shape)
        # Linear transformation of outcome 
        out = self.dropout(self.proj(out))
        return out
    
class Head(nn.Module):
    """
    One head of self-attention
    """
    def __init__(self, head_size, mask=False):
        super().__init__()
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CTX, CTX).to(device)).to(device))
        self.dropout = nn.Dropout(DROPOUT)
        self.mask = mask

    def forward(self, x):
        
        idx = x[0]
        mask = x[1]
      
        B, T, C = idx.shape
        # print('Head')
        # print(x.shape)

        key = self.key(idx).to(device)        # (B,T,C)
        query = self.query(idx).to(device)    # (B,T,C)

        # compute attention scores (affinities)
        # note scores divided by square root of channels to de-sharpen values for softmax later

        # match query against every key
        weights = query @ key.transpose(-2, -1).to(device) * (C ** -0.5) # (B,T,C) @ (B,C,T) -> (B,T,T) 
        
        # mask to ignore padding
        """
        ARGH !!!!!!!!!
        weights = weights.masked_fill(mask[:B, :T] == 0, float('-inf')).to(device)
        """
        # optional mask to ignore future tokens
        if self.mask:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')).to(device) # (B,T,T)

        # take softmax to determine each token's importance relative to any abritrary token
        weights = F.softmax(weights, dim=-1).to(device) # (B,T,T)
        # print(weights.shape)
        weights = self.dropout(weights)

        # weighted aggregation of the values
        v = self.value(idx).to(device) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        # print(out.shape)
        return out
    
class CrossAttention(nn.Module):
    """
    Multiple heads of cross-attention in parallel
    """
    def __init__(self, divided_head_size, mask=False):
        super().__init__()
        self.heads = nn.ModuleList([CrossHead(divided_head_size, mask) for _ in range(HEADS)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, memory):

        out = torch.concat([h.forward(x, memory) for h in self.heads], dim=-1)
    
        # Linear transformation of outcome 
        out = self.dropout(self.proj(out))
        return out
    
class CrossHead(nn.Module):
    """
    One head of masked self-attention
    """
    def __init__(self, head_size, mask):
        super().__init__()
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CTX, CTX).to(device)).to(device))
        self.dropout = nn.Dropout(DROPOUT)
        self.mask = mask

    def forward(self, x, memory):
        
        B, T, C = x.shape
        # print('Head')
        # print(x.shape)

        # Queries from previous block
        query = self.query(x).to(device)    # (B,T,C)

        # Keys from encoder 
        key = self.key(memory).to(device)  # (B,T,C)
        
        # compute attention scores (affinities)
        # note scores divided by square root of channels to de-sharpen values for softmax later

        # match query against every key
        weights = query @ key.transpose(-2, -1).to(device) * (C ** -0.5) # (B,T,C) @ (B,C,T) -> (B,T,T) 
        # print(weights.shape)
       
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
    
if __name__ == '__main__':
    train_example = "What is the worst customer service experience you have ever had? "
    train_example += '<|endoftext|>'
    idx = enc.encode(train_example, allowed_special={"<|endoftext|>"})
    idx += [50257 for _ in range(CTX - len(idx))]
    
    # Make B x T tensor
    test_batch = torch.tensor([idx for i in range(BATCH_SIZE)]).to(device)

    # Make the mask based on padding in tensor
    src_mask = torch.tensor([[0 if token == 50257 else 1 for token in tensor] for tensor in test_batch])

    class TestEncoder(nn.Module):

        def __init__(self):
            super().__init__()
            
            self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
            self.position_embedding_table = PositionalEncoding(EMBED_DIM, DROPOUT, CTX).to(device)
            self.encoder = nn.Sequential(*[Encoder() for _ in range(NX)])

        def forward(self, x, targets=None):
            
            idx = x[0]
            mask = x[1]

            tok_enb = self.token_embedding_table(idx) # B, T, C
            pos_enb = self.position_embedding_table(torch.transpose(tok_enb, 0, 1).to(device)) # T, B, C
            idx = torch.transpose(pos_enb, 1, 0).to(device) # B, T, C

            # Feed into encoder
            enc_out = self.encoder((idx, mask)) # -> B, T, C

            return enc_out

    model = TestEncoder().to(device)
    logits, mask = model((test_batch, src_mask))
    # print(logits.shape)

    model2 = Model().to(device)
    logits, loss = model((test_batch, src_mask))
    # print(logits.shape)
    bruh = torch.tensor([idx]).to(device)
    wtf = model2.generate_fixed(bruh, src_mask, 20)
    whoa = wtf[0][200:]
    print(enc.decode(whoa.tolist()))