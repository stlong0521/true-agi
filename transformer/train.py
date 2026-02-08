import torch
import torch.nn.functional as F
import requests
import os

# Hyperparameters for data loading
batch_size = 32
block_size = 64
train_split = 0.9

# Training hyperparameters
max_iters = 1000
eval_interval = 200
learning_rate = 3e-4
eval_iters = 100

# Device configuration
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Download the dataset if it doesn't exist
data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "./data/tinyshakespeare/input.txt"

if not os.path.exists(data_path):
    print(f"Downloading dataset from {data_url}...")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    response = requests.get(data_url)
    with open(data_path, "w") as f:
        f.write(response.text)

# Read the data
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of dataset in characters: {len(text)}")

# All unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_split * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x)
            # Reshape logits and targets for cross entropy
            batch_size, block_size, vocab_size = logits.shape
            logits = logits.view(batch_size * block_size, vocab_size)
            targets = y.view(batch_size * block_size)
            loss = F.cross_entropy(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    from transformer import Transformer

    # Model hyperparameters
    d_model = 128
    n_heads = 4
    d_ff = 512
    n_layers = 4

    # Instantiate the model
    model = Transformer(vocab_size, d_model, n_heads, d_ff, n_layers, block_size)
    model.to(device)
    print(f"Model instantiated and moved to {device}")

    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for iter in range(max_iters):
        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits = model(xb)
        batch_size, block_size, vocab_size = logits.shape
        logits = logits.view(batch_size * block_size, vocab_size)
        targets = yb.view(batch_size * block_size)
        loss = F.cross_entropy(logits, targets)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)  # Set all gradients to zero at each iteration to prevent gradient accumulation
        loss.backward()  # Compute gradients by backpropagating the loss
        optimizer.step()  # Update weights using the computed gradients

    # Final evaluation
    losses = estimate_loss(model)
    print(f"Final step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    print("Generating sample text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with a single 'zero' / newline token
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    print('----')
