# True AGI

Playground to test some crazy AGI ideas, starting with a character-level Transformer model.

## Setup Instructions

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

### 1. Install uv

If you don't have `uv` installed, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Restart your terminal to allow the installation to take effect.

### 2. Install Dependencies

Install the required packages and sync the environment (including the desired python version):

```bash
uv sync
```

## Running the Code

### Training and Data Preprocessing

To run the training script using `uv run`:

```bash
uv run python transformer/train.py
```

### Running Tests

To run the transformer unit tests:

```bash
uv run python transformer/test_transformer.py
```
