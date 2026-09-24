class Config:
    # tokenizer
    vocab_size = 2000

    # model — smaller for fast CPU training
    d_model = 128
    n_heads = 4
    n_layers = 4
    max_seq_len = 128

    # training
    batch_size = 8
    epochs = 1
    max_lr = 3e-4
    warmup_steps = 100
    grad_clip = 1.0
    device = "cuda"  # change to "cpu" if no GPU available

    # paths
    data_path = "data.txt"
    tokenizer_path = "bpe_tokenizer.json"
    checkpoint_path = "checkpoint.pt"