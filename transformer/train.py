import torch
import requests
import os

# Hyperparameters for data loading
batch_size = 32
block_size = 64
train_split = 0.9

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
    return x, y

if __name__ == "__main__":
    from transformer import Transformer

    # Model hyperparameters
    d_model = 128
    n_heads = 4
    d_ff = 512
    n_layers = 4

    # Instantiate the model
    model = Transformer(vocab_size, d_model, n_heads, d_ff, n_layers, block_size)
    print(f"Model instantiated with vocab_size={vocab_size}, d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, n_layers={n_layers}, block_size={block_size}")

    # Simple verification
    xb, yb = get_batch('train')
    print('inputs shape:', xb.shape)
    print('targets shape:', yb.shape)

    # Forward pass
    logits = model(xb)
    print('logits shape:', logits.shape)

    # Sanity check: logits shape should be (batch_size, block_size, vocab_size)
    expected_shape = (batch_size, block_size, vocab_size)
    if logits.shape == expected_shape:
        print("Success: Logits have the expected shape!")
    else:
        print(f"Error: Logits have shape {logits.shape}, expected {expected_shape}")

    print('----')
