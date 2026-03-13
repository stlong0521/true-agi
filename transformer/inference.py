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
    itos = metadata['itos']
    
    # Decoder function
    # Note: torch.save might save keys as strings if it was a JSON-like object, but here it's a dict.
    # However, when loading back, sometimes integer keys in dicts remain integers. 
    # Let's be robust:
    itos = {int(k): v for k, v in itos.items()}
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return model, decode

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = './model/checkpoint.pt'
    
    try:
        model, decode = load_model(model_path, device)
        print(f"Model loaded successfully from {model_path}")
        
        # Generate some text
        context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with a single 'zero' / newline token
        generated_chars = model.generate(context, max_new_tokens=100)[0].tolist()
        print("\nGenerated text:")
        print("--------------------------")
        print(decode(generated_chars))
        print("--------------------------")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run train.py first to save the model.")
