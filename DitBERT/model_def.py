"""Defines the DitBERT model and Masked LM task helper
"""

import math
import torch
from torch import nn
from torch import Tensor

import tabulate
import tokenizers


###############################################
###  Define Pytorch Model                   ###
###############################################

class TransformerEmbeddingModule(nn.Module):

    def __init__(self, d_model, vocab_size, max_seq_length, padding_idx, initilize_positional_with_sinusoids=True):
        """
        An embedding module for Transformer models, creating an embedding for each token and a positional embedding for
        the position in the sequence. The result is layer-normalized before being returned.

        Parameters
        ----------
        d_model : int
            The dimension of the token embeddings.
        vocab_size : int
            The size of the vocabulary.
        max_seq_length : int
            The maximum length of the input sequence.
        padding_idx : int
            The index of the padding token.
        initilize_positional_with_sinusoids : bool, optional, default=True
            Whether to initialize the positional embeddings with sinusoidal functions.

        Attributes
        ----------
        embedding : torch.nn.Embedding
            The token embedding module.
        pos_embedding : torch.nn.Embedding
            The positional embedding module.
        norm : torch.nn.LayerNorm
            The layer normalization module.
        """
        super(TransformerEmbeddingModule, self).__init__()

        # save parameters
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.padding_idx = padding_idx
        self.initilize_positional_with_sinusoids = initilize_positional_with_sinusoids


        # This is the embedding for the actual tokens
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # This is the positional embedding
        if initilize_positional_with_sinusoids:
            pe_init = self.get_sinusoidal_position_embeddings(max_len=max_seq_length, d_model=d_model)
            self.pos_embedding = nn.Embedding.from_pretrained(pe_init, freeze=False)
            # Change to freeze=True if you want to keep them fixed;
            # or freeze=False to just initalize with sine and cosine, but still train them afterwards
        else:
            self.pos_embedding = nn.Embedding(max_seq_length, d_model)

        self.norm = nn.LayerNorm(d_model)


    def get_sinusoidal_position_embeddings(self, max_len, d_model):
        """
        Calculate the sinusoidal position embeddings for a given maximum sequence length and embedding dimension.

        Parameters
        ----------
        max_len : int
            The maximum length of the sequence.
        d_model : int
            The dimension of the token embeddings.

        Returns
        -------
        torch.Tensor
            The sinusoidal position embeddings tensor.
            Shape: [max_len, d_model]

        """
        """Calculates the sinusoidal position embeddings for max_len of the sequence and d_model embedding dimension.
        Should be used to initalize a pytorch nn.Embedding class.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


    def forward(self, x):
        """
        Forward pass of the Transformer embedding module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing token indices.
            Shape: [batch size, sequence length]

        Returns
        -------
        torch.Tensor
            The output tensor containing the combined token and positional embeddings.
            Shape: [batch size, sequence length, d_model]
        """
        # Number the sequence with their respective positon for the positonal_embedding, then expand again to the batch nr
        # position_ids = torch.arange(x.size(1), dtype=torch.int, device=x.device).repeat(x.size(0), 1)
        # The below might work better than the above
        position_ids = torch.arange(x.size(1), dtype=torch.int, device=x.device).unsqueeze(0).expand_as(x)
        embedding = self.embedding(x) + self.pos_embedding(position_ids)
        return self.norm(embedding)



class DitBERTModel(nn.Module):

    def __init__(self, single_encoder_layer_params: dict, encoder_stack_params: dict,
        vocab_size: int, max_seq_length: int, padding_idx: int, initilize_positional_with_sinusoids: bool = True):
        """
        Initialize a DitBERTModel.

        Parameters
        ----------
        single_encoder_layer_params : dict
            The parameters for the single encoder layer.
        encoder_stack_params : dict
            The parameters for the encoder stack.
        vocab_size : int
            The size of the vocabulary.
        max_seq_length : int
            The maximum sequence length.
        padding_idx : int
            The index to be used for padding.
        initilize_positional_with_sinusoids : bool, optional
            Whether to initialize positional encoding with sinusoids, by default True.
        """
        super(DitBERTModel, self).__init__()

        # save parameters
        self.single_encoder_layer_params = single_encoder_layer_params
        self.encoder_stack_params = encoder_stack_params
        d_model = single_encoder_layer_params['d_model']
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.padding_idx = padding_idx
        self.initilize_positional_with_sinusoids = initilize_positional_with_sinusoids

        # Initialize the actual transformer model
        encoder_layer = nn.TransformerEncoderLayer(**single_encoder_layer_params)
        self.main_transformer = nn.TransformerEncoder(encoder_layer, **encoder_stack_params)

        self.full_embedding = TransformerEmbeddingModule(
            d_model=d_model, vocab_size=vocab_size, max_seq_length=max_seq_length, padding_idx=padding_idx,
            initilize_positional_with_sinusoids=initilize_positional_with_sinusoids
        )

        self.norm = nn.LayerNorm(d_model)
        self.to_token_map = nn.Linear(d_model, vocab_size, bias=False)

        # self.init_weights()


    def forward(self, src: Tensor) -> Tensor:
        """
        Forward propagation.

        Parameters
        ----------
        src : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The output tensor.
        """
        x = self.full_embedding(src)
        src_padding_mask = (src == self.padding_idx)
        out = self.main_transformer(src=x, src_key_padding_mask=src_padding_mask)
        out = self.norm(out)
        out = self.to_token_map(out)
        # return F.log_softmax(y, dim=-1) # -> for use with NLLLoss
        return out



###############################################
###  Define Tokenizer                       ###
###############################################

# python mapping dict
remap_dict_1 = {
    '„' : '"', # fix non-aligned beginnings -> no, don't because it screwed up more than it helped
    '“' : '"', # fix non-aligned beginnings
    '\u00a0' : ' ', # non-breaking white space
    '\u202f' : ' ', # narrow non-breaking white space
    'Ã¶' : 'ö', # german oe
    'Ã¼' : 'ü', # german ue
    'Ã¤' : 'ä', # german ae
}

remap_dict_2 = {
    '„'  : '"',
    '“'  : '"',
    '‟'  : '"',
    '”'  : '"',
    '″'  : '"',
    '‶'  : '"',
    '”'  : '"',
    '‹'  : '"',
    '›'  : '"',
    '’'  : "'",
    '′'  : "'",
    '′'  : "'",
    '‛'  : "'",
    '‘'  : "'",
    '`'  : "'",
    '–'  : '--',
    '‐'  : '-',
    '»'  : '"',
    '«'  : '"',
    '≪'  : '"',
    '≫'  : '"',
    '》' : '"',
    '《' : '"',
    '？' : '?',
    '！' : '!',
    '…'  : ' ... ',
    '\t' : ' ',
    '。' : '.', # chinese period
    '︰' : ':',
    '〜' : '~',
    '；' : ';',
    '）' : ')',
    '（' : '(',
    'ﬂ'  : 'fl', # small ligature characters
    'ﬁ'  : 'fi',
    '¶'  : ' ',
    chr(8211) : chr(45),
    '—'  : '-', #hypehen to normal minus
}


class PreCleansText: # (tokenizers.normalizers.Normalizer):
    def __init__(self, remap_dict_1, remap_dict_2):
        """An attempt to write a class to be added to the huggingface Tokenizers normalizers list.
        This did not work as this custom class could not be serialized and thus saved with the rest of the tokenizer"""
        self.remap_dict_1 = remap_dict_1
        self.remap_dict_2 = remap_dict_2

    def normalize(self, normalized: tokenizers.NormalizedString):
        for old, new in self.remap_dict_1.items():
            normalized.replace(old, new)
        for old, new in self.remap_dict_2.items():
            normalized.replace(old, new)

        # remove double spaces (there is not find function available for NormalizedString)
        for i in range(10):
            normalized.replace('  ', ' ')


def pre_cleanse_text(text):
    """Precleanses the text for the tokenizer"""
    for old, new in remap_dict_1.items():
        text = text.replace(old, new)
    for old, new in remap_dict_2.items():
        text = text.replace(old, new)

    # remove double spaces
    while text.find('  ') >= 0:
        text = text.replace('  ', ' ').replace('  ', ' ')

    return text


# Initialize tokenizer
tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())

tokenizer.normalizer = tokenizers.normalizers.Sequence([
    # tokenizers.normalizers.Normalizer.custom(PreCleansText(remap_dict_1=remap_dict_1, remap_dict_2=remap_dict_2)),
    tokenizers.normalizers.NFKC(),
    tokenizers.normalizers.StripAccents(), # StripAccents -> destorys German Umlaute, but for English that should be ok
])

tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
    tokenizers.pre_tokenizers.Digits(individual_digits = False), # separate digits, don't split individual ones
    tokenizers.pre_tokenizers.ByteLevel(),
])

tokenizer.decoder = tokenizers.decoders.ByteLevel()





###############################################
###  Helper Functions                       ###
###############################################


def count_parameters(model):
    """
    Counts the number of trainable parameters of a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which the number of trainable parameters will be counted.

    Returns
    -------
    int
        The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_count_table(model):
    """
    Displays a formatted table containing the number of trainable parameters
    for each named element of a PyTorch model, along with the total count.

    This function uses the `tabulate` library to create a table that lists
    the number of trainable parameters in each module of the provided PyTorch
    model. The table is formatted with two columns: 'Module' and 'Parameters'.
    It displays the module's name in the 'Module' column and the number of
    trainable parameters in the 'Parameters' column.

    The function also calculates the total number of trainable parameters
    in the model and appends this information to the table. The table is
    printed to the console.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which the parameter count table should be generated.

    Returns
    -------
    None
        The function does not return any value; it prints the table to the console.

    Requires
    --------
    tabulate : Python package
        The `tabulate` package is required to format and print the table.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    >>> parameter_count_table(model)
    """
    # table = PrettyTable(["Modules", "Parameters"])
    table = [ ["Module", "Parameters"] ]
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.append([str(name), params])
        total_params += params
    table.append(tabulate.SEPARATING_LINE)
    table.append(["TOTAL", total_params])
    print(tabulate.tabulate(table, headers="firstrow"))
    print("")


def adjust_model_dropout(model, new_dropout, verbose=1):
    """
    Adjusts the dropout values of all Dropout layers in a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model whose dropout values need to be adjusted.
    new_dropout : float
        The new dropout value to be applied to all Dropout layers in the model.
    verbose : int, optional, default=1
        If 1, prints the changes in dropout values for each Dropout layer;
        if 0, runs silently without printing any output.

    Returns
    -------
    None
        The function modifies the model in-place and does not return any value.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(10, 20), nn.Dropout(0.5), nn.ReLU(), nn.Linear(20, 1))
    >>> adjust_model_dropout(model, 0.25, verbose=1)
    Changing >>1<<'s current dropout value of 0.5 to new value 0.25
    """
    for idx, m in enumerate(model.named_modules()):
        path = m[0]
        component = m[1]
        if isinstance(component, nn.Dropout):
            if verbose: print(f"Changing >>{m}<<'s current dropout value of {component.p} to new value {new_dropout}")
            component.p = new_dropout


def mask_tokens(input_tensor, mask_token_id, nr_special_tokens, vocab_size, mask_prob=0.15):
    """Masks tokens of a tensor according to the masked language model task,
    where in the original BERT and RoBERTa implementations
    15% of tokens were masked, of which
        - 80% were replaced by [MASK] token,
        - 10% were replaced by a random token (where we ignore the special tokens, at the beginning of the vocab)
        - 10% are not replaced but just inputed into the loss as well (i.e. they have to be predicted as not being random)

    Returns:
        1. The modified tensor (to be used as input for the transformer during (pre-)training)
        2. The indices used for masking
        3. The original values (to be held against the predictions in the loss calculation)

    Example:

        ~~~~~python
        batch_size = 32
        sequence_length = 128
        input_tensor = torch.randint(0, 10000, (batch_size, sequence_length))
        mask_token_id = 123  # You can replace this with the id of the [MASK] token in your tokenizer
        nr_special_tokens = 5 # The number of special tokens (i.e. length of the special tokens list)
        vocab_size = 10000

        masked_tensor, masked_indices, original_values = mask_tokens(input_tensor, mask_token_id, nr_special_tokens, vocab_size)
        ~~~~~
    """
    # Set the device of the tensor
    device = input_tensor.device

    # Create a boolean mask where the value is True with probability `mask_prob`
    mask = torch.rand(input_tensor.shape, device=device) < mask_prob

    # Get the indices where the mask is True
    mask_indices = mask.nonzero(as_tuple=True)

    # Remember the original values
    original_values = input_tensor[mask_indices].clone().long() # has to be long for cross entropy loss function

    # Replace 80% of the selected tokens with the [MASK] token
    n_mask = int(0.8 * len(mask_indices[0]))
    input_tensor[mask_indices[0][:n_mask], mask_indices[1][:n_mask]] = mask_token_id

    # Replace 10% of the selected tokens with random tokens
    n_random = int(0.1 * len(mask_indices[0]))
    random_tokens = torch.randint(nr_special_tokens, vocab_size, (n_random,), device=device)
    input_tensor[mask_indices[0][n_mask:n_mask + n_random], mask_indices[1][n_mask:n_mask + n_random]] = random_tokens

    # The remaining 10% of the selected tokens will not be changed

    return input_tensor, mask_indices, original_values


def get_lr(optimizer):
    """
    Get the learning rate of an optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The PyTorch optimizer instance whose learning rate needs to be retrieved.

    Returns
    -------
    float
        The learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']



####################################################
## Define Data Loader                             ##
####################################################

class LM_FileLoader:

    def __init__(self, file_list, tokenizer, max_sequence_length, sub_batch_size, vocab_size, open_starting_position=None):
        """Initialize the LM_FileLoader for language modeling.

        Args:
            file_list (list): A list of tuples (or 2-element lists) containing the number of opens to initialize and file paths.
            tokenizer (Tokenizer): A tokenizer object to tokenize the text data.
            max_sequence_length (int): The maximum sequence length for the output tensor.
            sub_batch_size (int): The size of the sub-batch.
            vocab_size (int): The size of the vocabulary, needed for random testing.
            open_starting_position (list): A list of integers representing the starting line numbers for each file in the file_list.
                                           If None, starts from the beginning of each file.

        """
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.sub_batch_size = sub_batch_size
        self.vocab_size = vocab_size

        self.start_token_id = tokenizer.token_to_id('[START]')
        self.end_token_id = tokenizer.token_to_id('[END]')

        nr_total_opens = sum([l[0] for l in file_list])
        assert nr_total_opens == sub_batch_size, f"Error: Sum of opens expected from file_list {nr_total_opens} did not match sub_batch_size {sub_batch_size}"

        self.open_list = []
        self.open_list_opened_file = []  # to remember which file was opened
        self.current_line_positions = [0] * nr_total_opens

        if open_starting_position is None:
            print("No open_starting_position provided. Initializing files with offset.")
            position_idx = 0
            for nr_opens, file in self.file_list:

                # determine number of lines:
                with open(file, 'r', encoding='utf-8') as ff:
                    nr_lines = len([0 for _ in ff])

                # calculate offset (in terms of multiple opens we shift the opens equally along the file)
                calc_offset = (nr_lines/nr_opens)
                print(f"Calculated offset for file {file.name}: {calc_offset}")

                for i in range(nr_opens):
                    ff = open(file, 'r', encoding='utf-8')
                    for off_idx in range(round(i*calc_offset)):
                        ff.readline()
                        self.current_line_positions[position_idx] += 1
                    self.open_list.append(ff)
                    self.open_list_opened_file.append(file)
                    position_idx += 1
            print(f"Nr initialized opens: {len(self.open_list)}")
        else:
            print("Starting positions provided:")
            position_idx = 0
            for nr_opens, file in self.file_list:
                for _ in range(nr_opens):
                    print(f"\tinitilizing {file.name} to position {open_starting_position[position_idx]}")
                    ff = open(file, 'r', encoding='utf-8')
                    for _ in range(open_starting_position[position_idx]):
                        ff.readline()
                        self.current_line_positions[position_idx] += 1
                    self.open_list.append(ff)
                    self.open_list_opened_file.append(file)
                    position_idx += 1

        self.memory_bank = [[] for _ in range(len(self.open_list))]



    def get_batch(self):
        """Get a batch of tokenized sequences for training.

        Returns:
            torch.Tensor: A tensor of shape (sub_batch_size, max_sequence_length) containing tokenized sequences.
        """
        out_tensor = []
        for idx in range(len(self.open_list)):
            while len(self.memory_bank[idx]) < self.max_sequence_length:
                # Load more data to the memory bank, until we have at least max-sequence-length
                line = self.open_list[idx].readline()
                if line == '':
                    self.check_and_reopen_file(idx)
                    line = self.open_list[idx].readline()
                self.current_line_positions[idx] += 1
                self.memory_bank[idx].extend(self.tokenizer.encode(line).ids)

            tensor_line = [self.start_token_id] + self.memory_bank[idx][:self.max_sequence_length].copy() + [self.end_token_id]
            # tensor_line = self.memory_bank[idx][:self.max_sequence_length].copy() + [
            out_tensor.append(torch.Tensor([tensor_line]).clone())
            del self.memory_bank[idx][:self.max_sequence_length] # delete elements for next run
        return torch.cat(out_tensor, dim=0).long()

    def get_random_batch(self):
        """Get a random batch of tokenized sequences for testing.

        Returns:
            torch.Tensor: A tensor of shape (sub_batch_size, max_sequence_length) containing random tokenized sequences.
        """
        return torch.randint(low=0, high=self.vocab_size, size=(self.sub_batch_size, self.max_sequence_length)).long()

    def get_line_positions(self):
        """
        Get the current line positions for each open file.

        Returns:
            list: A list of integers representing the current line positions for each open file.
        """
        return self.current_line_positions

    def set_line_positions(self, line_positions):
        """
        Set the current line positions for each open file.

        Args:
            line_positions (list): A list of integers representing the new line positions for each open file.
        """
        for idx, new_position in enumerate(line_positions):
            current_position = self.current_line_positions[idx]
            if new_position == current_position:
                continue
            elif new_position > current_position:
                for _ in range(new_position - current_position):
                    self.open_list[idx].readline()
            else:
                self.open_list[idx].seek(0)
                for _ in range(new_position):
                    self.open_list[idx].readline()
            self.current_line_positions[idx] = new_position


    def check_and_reopen_file(self, open_idx):
        """Check if the end of the file has been reached and reopen it if needed.

        Args:
            open_idx (int): Index of the open file in self.open_list.
        """
        # Check if end of file is reached.
        file_pos = self.open_list[open_idx].tell()
        self.open_list[open_idx].seek(0, 2)  # Move file pointer to the end of the file (2=ending of file in the second argument).
        end_pos = self.open_list[open_idx].tell()

        if file_pos == end_pos:
            # End of file reached, close and reopen the file.
            self.open_list[open_idx].close()
            self.open_list[open_idx] = open(self.open_list_opened_file[open_idx], 'r', encoding='utf-8')
            self.current_line_positions[open_idx] = 0

        else:
            # Reset file pointer to its original position.
            self.open_list[open_idx].seek(file_pos)
