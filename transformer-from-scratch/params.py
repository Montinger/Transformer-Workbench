
nr_epochs = 30 # only counting proper epochs; +1 warming epoch (0) with linear lr increase, which is automatically added
max_nr_tokens = 200 # after this length the tokens are simply cut
bpe_vocab_size = 32_000
batch_nr_tokens = 25_000 # The number of tokens per batch (cuts of after source or target reaches this amount of tokens)
padding_token_id = 1 # The id of the [pad] token from the tokenizer

# params for training
train_batch_file = "train_batches-25000.pkl" # which file to use for training
train_trafo_type = 'own' # either 'ref', 'mask' or 'own'

transformer_params_base = {
    "d_model": 512,
    "nhead": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "dim_feedforward": 2048,
    "dropout": 0.1,
}

# for testing
transformer_params_mini = {
    "d_model": 256, # 512,
    "nhead": 8, # 8,
    "num_encoder_layers": 3, # 6,
    "num_decoder_layers": 3, #6,
    "dim_feedforward": 1028, # 2048,
    "dropout": 0.1,
}

transformer_params = transformer_params_base
