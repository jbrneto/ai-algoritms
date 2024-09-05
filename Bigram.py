import torch
import torch.nn as nn
from torch.nn import functional as F

# https://github.com/Infatoshi/fcc-intro-to-llms
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    # T: timestep is how much of the index does the embedding saw
    # [['h','e','l',0.,0.]
    #  ['h','e','l','l',0.]
    #  ['h','e','l','l','o']]
    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            
            # focus only on the last timestep
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)

            # append to sequence
            index = torch.cat((index, index_next), dim=1)
        return index


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# read file
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)

# encode data
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)

# split batches
block_size = 8 # block of chars
batch_size = 4 # batch of blocks
n = int(0.8*len(data))

train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

model = Bigram(vocab_size)
m = model.to(device)

# Evaluation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
max_iters = 1000
learning_rate = 3e-4
eval_iters = 250
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters):
    if it % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {it}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

# Prediction
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)
