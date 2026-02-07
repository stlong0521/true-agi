import torch
import requests
import os

# Hyperparameters for data loading
batch_size = 32
block_size = 64  # context length
train_split = 0.9

# Download the dataset if it doesn't exist
data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "../data/tinyshakespeare/input.txt"

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
    # Simple verification
    xb, yb = get_batch('train')
    print('inputs:')
    print(xb.shape)
    print(xb)
    print('targets:')
    print(yb.shape)
    print(yb)

    print('----')

    for b in range(1): # print the first batch entry
        for t in range(3): # print first 3 time steps
            context = xb[b, :t+1]
            target = yb[b,t]
            print(f"when input is {context.tolist()} the target: {target}")
