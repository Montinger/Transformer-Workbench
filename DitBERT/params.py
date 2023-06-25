"""This script contains the fine tuning parameters for the DitBERT model and later fine tuning

Some of the parameters are written in separate dicts for easy parsing to the constructors with **my_params
"""
import pathlib


# Values are from RoBERTa Base
single_encoder_layer_params = {
    "d_model": 768,
    "nhead": 12,
    "dim_feedforward": 3072,
    "dropout": 0.0, # Original RoBERTa-base 0.1
    "activation": 'gelu',
    "batch_first": True,
    "norm_first": True, # should be more stable
}
# for usage like: encoder_layer = nn.TransformerEncoderLayer(**single_encoder_layer_params)

encoder_stack_params = {
    "num_layers": 12,
}
# for usage like: transformer_encoder = nn.TransformerEncoder(encoder_layer, **encoder_stack_params)


adamW_parameters = {
    "lr": 0.001, # this should be the peak learning rate. Original RoBERTa-base 0.0006, but the cramming paper suggests this maximum
    "betas": (0.9, 0.98),
    "eps": 1e-06,
    "weight_decay": 0.01,
}
# for usage like: optimizer = torch.optim.AdamW(model.parameters(), **adamW_parameters)

# The optimizer for fine-tuning on the GLUE Benchmark. Also see finetune_params.
adamW_parameters_finetune = {
    "lr": 2e-5,
    "betas": (0.9, 0.98),
    "eps": 1e-06,
    "weight_decay": 0.1,
}

other_params = {
    "nr_warmup_steps": 25_000,
    "nr_total_steps": 75_000, # normal steps will only start counting after warmup finishes (Roberta was 500_000)
    "batch_size": 3_200, # as nr of sequences with max_sequence_length. Original RoBERTa-base 8_000
    "sub_batch_size": 32, # The batch is processed in parts of this size. With original RoBERTa 12GB can only handle ~4
    "gradient_clipping": 0.5, # original paper used no gradient clipping
    "bpe_vocab_size": 32_768, # Original RoBERTa-base 50_000
    "max_sequence_length": 128, # nr of tokens. Original RoBERTa-base 512. This count is without [START] and [END] token
    "special_tokens": ['[PAD]', '[START]', '[END]', '[MASK]', '[DOC]'],
    "padding_idx": 0,
    "save_after_steps": 1_000,
}
# learning rate will always be linearly decayed


training_files_input_path = pathlib.Path("/data_ssd/tmp_text_preparation/cleansed-parsed/")
# The first number specifies how many open instances will be created of the file.
# The sum of these values should match sub_batch_size
# The values were roughly assigned based on file size
training_files_list = [
    [ 1, pathlib.Path(training_files_input_path / "cc_news_cleansed.txt")],
    [ 1, pathlib.Path(training_files_input_path / "edgar_cleansed.txt")],
    [ 4, pathlib.Path(training_files_input_path / "news_raw_parsed_output_en.txt")],
    [ 5, pathlib.Path(training_files_input_path / "enwikisource-20230101-cleansed.txt")],
    [ 9, pathlib.Path(training_files_input_path / "gutenberg_cleansed.txt")],
    [12, pathlib.Path(training_files_input_path / "enwiki-20230101.txt")],
]

# The fine-tuning parameters for the GLUE Benchmark. Also see adamW_parameters_finetune.
finetune_params = {
    "nr_warmup_epochs": 2,
    "nr_epochs": 8, # not counting the warm-up epoch
    "batch_size": 32,
    "new_dropout": 0.1 # to what to change the dropout value for fine-tuning
}
