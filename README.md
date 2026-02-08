# True AGI

Playground to test some crazy AGI ideas, starting with a character-level Transformer model.

## Setup Instructions

This project uses a virtual environment to manage dependencies.

### 1. Create a Virtual Environment

If you haven't already created one, run the following command in the root directory:

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

Activate the environment based on your operating system:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Code

### Training and Data Preprocessing

To run the training script (which currently downloads the Tiny Shakespeare dataset, preprocesses it, and performs a forward pass check):

```bash
cd transformer
python train.py
```

### Running Tests

To run the transformer unit tests:

```bash
cd transformer
python test_transformer.py
```
