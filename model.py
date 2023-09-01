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
VOCAB_SIZE = 50257 + 1 + 1 # 51000
PADDING = 50257 
START_TOKEN = 50258
LR = 6e-4 # 3e-4
DROPOUT = 0.1
HEADS = 8
NX = 8
LR = 1e-5 # 6e-4
BATCH_SIZE = 14 # 8
CTX = 200 # 52
EMBED_DIM = 512 # 128
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
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding_table = PositionalEncoding(EMBED_DIM, DROPOUT, CTX).to(device)
        self.encoder = nn.Sequential(*[Encoder() for _ in range(NX)])
        self.decoder = nn.Sequential(*[Decoder() for _ in range(NX)])
        self.ln_final = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x, src_mask, predict, pred_mask) -> torch.tensor: 

        # add embedding to input tokens
        tok_enb = self.token_embedding_table(x) # B, T, C
        pos_enb = self.position_embedding_table(torch.transpose(tok_enb, 0, 1).to(device)) # T, B, C
        x = torch.transpose(pos_enb, 1, 0).to(device) # B, T, C

        # Feed logits and padding mask into encoder
        memory, _ = self.encoder((x, src_mask)) # -> B, T, C

        # add embedding to decoder tokens
        tok_enb_predict = self.token_embedding_table(predict)
        pos_enb_predict = self.position_embedding_table(torch.transpose(tok_enb_predict, 0, 1).to(device))
        predict = torch.transpose(pos_enb_predict, 1, 0).to(device)

        # Feed predicted tokens, padding mask and encoder outputs into decoder 
        x, _, _, _ = self.decoder((predict, memory, src_mask, pred_mask))  
        x = self.ln_final(x)

        # finally plug into language model head
        logits = self.lm_head(x) # B, T, vocab_size 

        return logits
    
    @torch.no_grad()
    def generate(self, x, src_mask) -> [int]:
        """
        Generates output from the model. Stops generating when end of text token generates.
        x: 1 x N tensor
        """
        
        output = []
        predict = torch.tensor([[PADDING for _ in range(CTX)] for _ in range(1)]).to(device)
        end = False
        failsafe = CTX + 50

        while True:
            
            # Make mask for decoder outputs 
            pred_mask = torch.tensor([[0 if token == PADDING else 1 for token in tensor] for tensor in predict]).to(device)

            # One step in model
            logits = self(x, src_mask, predict, pred_mask)

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
        self.ca = CrossAttention(divided_head_size, mask=True)
        self.ffwd = FeedForward(EMBED_DIM)
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ln3 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x: tuple):
        
        idx, memory, src_mask, pred_mask = x
        # Masked self-attention
        idx = self.sa((self.ln1(idx), pred_mask)) + idx

        # Cross-attention
        idx = self.ca(self.ln2(idx), memory, src_mask, pred_mask) + idx 
        
        # Computation with residual connection
        idx = self.ffwd(self.ln3(idx)) + idx
        
        # Return masks as well since this goes through multiple heads
        return (idx, memory, src_mask, pred_mask)
    
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
        # print("Outcome: ", out.shape)
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

        key = self.key(idx).to(device)        # (B,T,C)
        query = self.query(idx).to(device)    # (B,T,C)

        # mask to ignore padding tokens
        mask_extend = mask[:, :T, None] 
        # print(key)
        key = key.masked_fill(mask_extend == 0, 0).to(device)
        query = query.masked_fill(mask_extend == 0, 0).to(device)   

        # compute attention scores (affinities)
        # note scores divided by square root of channels to de-sharpen values for softmax later

        # match query against every key
        weights = query @ key.transpose(-2, -1).to(device) * (C ** -0.5) # (B,T,C) @ (B,C,T) -> (B,T,T) 

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

    def forward(self, x, memory, src_mask, pred_mask):

        out = torch.concat([h.forward(x, memory, src_mask) for h in self.heads], dim=-1)
    
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

    def forward(self, x, memory, src_mask):
        
        B, T, C = x.shape
        # print('Head')
        # print(x.shape)

        # Queries from previous block
        query = self.query(x).to(device)    # (B,T,C)

        # Keys from encoder 
        key = self.key(memory).to(device)  # (B,T,C)
        
        # compute attention scores (affinities)
        # note scores divided by square root of channels to de-sharpen values for softmax later
        
        # Apply memory mask (src mask) to keys
        mask_extend = src_mask[:, :T, None] 
        key = key.masked_fill(mask_extend == 0, 0).to(device)

        # match query against every key
        weights = query @ key.transpose(-2, -1).to(device) * (C ** -0.5) # (B,T,C) @ (B,C,T) -> (B,T,T) 
        # print(weights.shape)

        # optional mask to ignore future tokens
        if self.mask:
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
    
if __name__ == '__main__':
    train_example = "What is the worst customer service experience you have ever had? "
    train_example += '<|endoftext|>'
    idx = enc.encode(train_example, allowed_special={"<|endoftext|>"})
    idx += [PADDING for _ in range(CTX - len(idx))]
    
    # Make 1 x T tensor (only 1 example for testing)
    test_batch = torch.tensor([idx for i in range(BATCH_SIZE)]).to(device)

    # Make the mask based on padding in tensor
    src_mask = torch.tensor([[0 if token == PADDING else 1 for token in tensor] for tensor in test_batch]).to(device)

    """
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
    print(logits.shape)

    """

    model2 = Model().to(device)
    
    correct_output_example = "My worst customer experience? My worst customer experience would have to be the time an old man came in and started demanding to use a bathroom. "
    correct_output_example += '<|endoftext|>'
    correct_outputs = enc.encode(correct_output_example, allowed_special={'<|endoftext|>'})
    decoder_inputs = [START_TOKEN] + correct_outputs + [PADDING for _ in range(CTX - len(correct_outputs) - 1)]
    correct_outputs += [PADDING for _ in range(CTX - len(correct_outputs))]
    outputs = torch.tensor([decoder_inputs for _ in range(BATCH_SIZE)]).to(device)
    targets = torch.tensor([correct_outputs for _ in range(BATCH_SIZE)]).to(device) 
    pred_mask = torch.tensor([[0 if token == PADDING else 1 for token in tensor] for tensor in outputs]).to(device) 

    print(outputs.shape, targets.shape, pred_mask.shape)

    visualization = ""
    
    logits = model2(test_batch, src_mask, targets, pred_mask) 

    # Textualization of logits
    cropped = logits[:, -1, :] 
    probs = F.softmax(cropped, dim=-1) # (B, C)
    x_next = torch.multinomial(probs, num_samples=1).to(device) # (B, 1)
    visualization += enc.decode(x_next.tolist()[0])
    print(visualization)

    # Loss calculation  
    batch_size, CTX, vocab_size = logits.shape
    logits = logits.view(batch_size * CTX, vocab_size)
    targets = targets.view(batch_size*CTX)
    loss = F.cross_entropy(logits, targets)

    print(loss)    

    # Generating from the model
    eval_batch = torch.tensor([idx for _ in range(1)]).to(device)
    eval_mask = torch.tensor([[0 if token == PADDING else 1 for token in tensor] for tensor in eval_batch]).to(device)
    thing = model2.generate(eval_batch, eval_mask)
    print(enc.decode(thing)[-1])
