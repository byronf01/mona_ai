import os 

# Third-party
import sentencepiece as spm
import torch
import torch.nn as nn 
from torch.nn import functional as F


N_EMBED = 384
HEADS = 6
N_LAYER = 6
device = "cuda:0" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 256
BATCH_SIZE = 64
LR = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
DROPOUT = 0.2

def get_batch(split, train_data, val_data):
    """
    Generates batch data of BATCH_SIZE of inputs x which is of BLOCK_SIZE and targets y
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,), device=device)
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]).to(device)
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

class Block(nn.Module):
    """
    Transformer block, communication followed by computation
    """
    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension
        # n_head: number of heads to use
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):

        # Communication with residual connection
        x = self.sa(self.ln1(x)) + x

        # Computation with residual connection
        x = self.ffwd(self.ln2(x)) + x
        return x
    


class BLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        # self.sa_heads = MultiHeadAttention(HEADS, N_EMBED // HEADS) 
        # self.ffwd = FeedForward(N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, HEADS) for _ in range(N_LAYER)])
        self.ln_final = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx and targets are both (B,T) tensor of integers, function feeds input activations 
        through network and predicts the output
        """
        B, T = idx.shape

        # embed tokens
        tok_enb = self.token_embedding_table(idx) # B, T, C

        # embed positions (positions labeled 0 thru block_size - 1)
        pos_enb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        # Apply multi-headed self-attention
        x = tok_enb + pos_enb # (B,T,C)
        x = self.blocks(x) 
        x = self.ln_final(x)

        # finally plug into language model head
        logits = self.lm_head(x) # B, T, vocab_size 

        if targets is None: loss = None
        else:
            # must convert tensor size for logits/targets for each token in vocab to have 
            # activation value between 0 and 1
            batch_size, block_size, vocab_size = logits.shape
            logits = logits.view(batch_size * block_size, vocab_size)
            targets = targets.view(batch_size*block_size)
            loss = F.cross_entropy(logits, targets)

        return (logits, loss)
    
    def generate(self, idx, max_new_tokens):
        """
        idx is the current context of characters in some batch, size is batch x block and 
        whatever is generated by feeding forward gets concatentated onto idx
        """
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens to prevent too much
            idx_crop = idx[:,-BLOCK_SIZE:]
            # get the predictions, ignore loss
            logits, _ = self(idx_crop)
            # focus only on the last element in time dimension
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1).to(device) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class Head(nn.Module):
    """
    One head of self-attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE).to(device)).to(device))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
  
        key = self.key(x).to(device)        # (B,T,C)
        query = self.query(x).to(device)    # (B,T,C)

        # compute attention scores (affinities)
        # note scores divided by square root of channels to de-sharpen values for softmax later

        # match query against every key
        weights = query @ key.transpose(-2, -1).to(device) * (C ** -0.5) # (B,T,C) @ (B,C,T) -> (B,T,T) 

        # optional mask to ignore future tokens
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')).to(device) # (B,T,T)

        # take softmax to determine each token's importance relative to any abritrary token
        weights = F.softmax(weights, dim=-1).to(device) # (B,T,T)
        weights = self.dropout(weights)

        # weighted aggregation of the values
        v = self.value(x).to(device) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        out = torch.concat([h(x) for h in self.heads], dim=-1)

        # Linear transformation of outcome 
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """
    Linear layer followed by a non-linearity
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)
    

if __name__ == '__main__':
    
    with open('test.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # set up encoding
    chars = set(text)
    if not os.path.exists('./m.model'): 
        spm.SentencePieceTrainer.train(f'--input=test.txt --model_prefix=m --vocab_size={len(chars)}')
    sp = spm.SentencePieceProcessor(model_file='./m.model')

    data = torch.tensor(sp.encode(text), dtype=torch.long, device=device)
    # print(data[:1000])

    n = int(0.9*len(data))
    train_data = data[:n]
    validation_data = data[n:]

    xb, yb = get_batch('train', train_data, validation_data)


    # Setting up the model
    m = BLM(len(chars))
    m.to(device)
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)

    print('Before: ')
    print(sp.decode_ids(m.generate(idx=torch.zeros((1,1),dtype=torch.long,device=device), max_new_tokens=1000)[0].tolist() ))
    
    # create torch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for iter in range(MAX_ITERS):

        # periodically evaluate loss on training and validation sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(m, train_data, validation_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, validation_data)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        

    print('After: ')
    print(sp.decode_ids(m.generate(idx=torch.zeros((1,1),dtype=torch.long,device=device), max_new_tokens=3000)[0].tolist() ))

# s