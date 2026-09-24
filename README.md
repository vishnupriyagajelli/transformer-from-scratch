# Transformer From Scratch

This is a small language model I built using PyTorch, without using Hugging Face or any other ready-made library.
I wanted to actually understand how things like attention and tokenizers work, so I coded each part myself.

## What I built

- A tokenizer (BPE) that breaks text into pieces and turns them into numbers
- Multi-head self-attention, written using basic matrix math
- RoPE (a way of telling the model the position of each word), used in models like LLaMA
- RMSNorm, a normalization layer used in modern models
- A training loop with mixed precision, gradient clipping, and a learning rate warmup
- KV-caching, which makes text generation faster

## Files

- `config.py` - all the settings (model size, learning rate, etc.)
- `tokenizer.py` - the tokenizer
- `model.py` - the actual transformer model
- `dataset.py` - turns text into training data
- `train.py` - trains the model
- `generate.py` - generates text using the trained model

## How to run it

Install the requirements:
pip install -r requirements.txt

Put some text in a file called `data.txt` in this folder.

Train the model:
python train.py

Generate text:
python generate.py

## Why I made this

Most projects just call an API or use a library like Hugging Face.
I wanted to build the actual pieces myself so I could understand what's really happening inside a transformer, instead of just using it as a black box.

## Note

This is a small model trained on a small amount of text, so the generated text won't be perfect.
The point of this project was learning how transformers work, not building something production-ready.
## How to run it

Install the requirements:
