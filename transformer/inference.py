import torch
from transformer import Transformer


def load_model(model_path, device):
    """
    Load the model and its metadata from a checkpoint.
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract hyperparameters
    hp = checkpoint['hyperparameters']
    
    # Instantiate the model
    model = Transformer(
        vocab_size=hp['vocab_size'],
        d_model=hp['d_model'],
        n_heads=hp['n_heads'],
        d_ff=hp['d_ff'],
        n_layers=hp['n_layers'],
        block_size=hp['block_size']
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extract metadata (character mappings)
    metadata = checkpoint['metadata']
    stoi = metadata['stoi']
    itos = metadata['itos']
    
    # Ensure keys are integers for itos
    itos = {int(k): v for k, v in itos.items()}

    # Encoder and Decoder functions
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return model, encode, decode

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = './model/checkpoint.pt'
    
    try:
        model, encode, decode = load_model(model_path, device)
        print(f"Model loaded successfully from {model_path}")
        
        # Define a prompt
        prompt = "ROMEO:"
        print(f"\nPrompt: {prompt}")

        # Encode the prompt
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)  # (1, len(prompt))

        # Generate some text
        generated_chars = model.generate(context, max_new_tokens=100)[0].tolist()
        print("\nGenerated text:")
        print("--------------------------")
        print(decode(generated_chars))
        print("--------------------------")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run train.py first to save the model.")
