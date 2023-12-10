"""This script contains the fine tuning parameters for the LoRA from Scratch RoBERTa model for fine tuning

The same values are used for each task.

The adamW parameters are written to a separate dict for easy parsing to the constructor.
"""

# The optimizer for fine-tuning on the GLUE and SQUAD Benchmark. Also see finetune_params.
adamW_parameters_finetune = {
    "lr": 4e-5,
    "betas": (0.9, 0.98),
    "eps": 1e-06,
    "weight_decay": 0.1,
}
# for usage like: optimizer = torch.optim.AdamW(model.parameters(), **adamW_parameters)


# The fine-tuning parameters for the GLUE and SQUAD Benchmark. Also see adamW_parameters_finetune.
finetune_params = {
    "nr_warmup_epochs": 2,
    "nr_epochs": 4, # 8, # not counting the warm-up epoch
    "batch_size": 16,
    "new_dropout": 0.1 # to what to change the dropout value for fine-tuning
}
