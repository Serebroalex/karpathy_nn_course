import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# hyperparameters 
batch_size = 64 # how many independent sequences do we process in parallel on every forward/backward pass?
block_size = 64 # what is the maximum context length for prediction?
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_heads = 8
n_layer = 8
dropout = .2

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt','r',encoding = 'utf-8') as f:
    text = f.read()

# unique characters occuring in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# mapping from characters to integers
s_to_i = {ch:i for i,ch in enumerate(chars)}
i_to_s = {i:ch for i,ch in enumerate(chars)}
encode = lambda s:[s_to_i[c] for c in s] # encoder takes in a string, outputs a list of integers
decode = lambda l:''.join([i_to_s[i] for i in l]) # decoder takes in a list of integers, outputs a string

# train/test split
data = torch.tensor(encode(text), dtype = torch.long)
n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generat a minibatch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # generate random positions from which to get the data 
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # sample a batch of data
            X, Y = get_batch('train')
            
            # evaluate loss
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out   


class Head(nn.Module):
    # one head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-.5 # (B, T, C) * (B, C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        wei = self.dropout(wei)

        # perform weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) ---> (B, T, C)

        return out

class MultiHeadAttention(nn.Module):
    # multiple parallel self-attention heads
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # outputs of each head concatenated over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    # feed forward nonlinear layer
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Transformer block
    def __init__(self, n_embd, n_heads):
        # n_embd: embedding dimension, n_head: the number of heads
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_heads = n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, ix, targets = None):
        B, T = ix.shape
        # ix and target are (B, T) tensors of integers
        tok_emb = self.token_embedding_table(ix) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        # apply self-attention blocks
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, ix, max_new_tokens):
        # ix is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop ix (the context) to the last block_size tokens
            ix_cond = ix[:, -block_size:]

            # get the predictions
            logits, loss = self(ix_cond) # logits - (B,T,C)
            # focus on the last time step
            logits = logits[:,-1,:] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # (B, C)
            # sample from the distribution
            ix_next = torch.multinomial(probs, num_samples = 1)
            # append sampled index to the running sequence
            ix = torch.cat((ix, ix_next), dim = 1) # (B, T + 1)
            
        return ix
    
model = TransformerLanguageModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

t0 = time.time()
for k in range(max_iters):

    # print loss once in a while
    if k % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {k}/{max_iters}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}, time elapsed {time.time() - t0:.4f} seconds")

    # sample a batch of data
    x_b, y_b = get_batch('train')
    
    # evaluate loss
    logits, loss = model(x_b, y_b)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
print(f"step {max_iters}/{max_iters}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}, time elapsed {time.time() - t0:.4f} seconds")

# generate from the mode
context = torch.zeros((1,1), dtype = torch.long, device = device)
output = decode(model.generate(context, max_new_tokens = 1000)[0].tolist())
print(output)
with open('Not_Shakespeare.txt', 'w') as f:
    f.write(output)
