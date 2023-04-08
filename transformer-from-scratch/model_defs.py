import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F



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



class TransformerModel(nn.Module):

    def __init__(self, use_transformer, transformer_params, vocab_size, max_seq_length, padding_idx, initilize_positional_with_sinusoids=True):
        """
        A Transformer model that supports using either the PyTorch built-in transformer or a custom 'from scratch' implementation of a transformer

        Parameters
        ----------
        use_transformer : str
            The type of transformer to use, either 'ref' or 'mask' for the built-in PyTorch transformer,
            or 'own' for a custom 'from scratch' implementation.
        transformer_params : dict
            A dictionary containing the parameters for the transformer model.
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
        main_transformer : torch.nn.Module
            The main transformer model.
        full_embedding : TransformerEmbeddingModule
            The token and positional embedding module.
        norm : torch.nn.LayerNorm
            The layer normalization module.
        to_token_map : torch.nn.Linear
            The linear layer to map the transformer output to the vocabulary size.
        """
        super(TransformerModel, self).__init__()

        # save parameters
        self.transformer_params = transformer_params
        d_model = transformer_params['d_model']
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.padding_idx = padding_idx
        self.initilize_positional_with_sinusoids = initilize_positional_with_sinusoids

        self.transformer_type = use_transformer
        # Initialize the actual transformer model
        if use_transformer in ["ref", "mask"]:
            self.main_transformer = torch.nn.Transformer(**transformer_params, batch_first=True, activation='gelu', norm_first=True)
        elif use_transformer == "own":
            self.main_transformer = TransformerBlockFromScratch(**transformer_params)
        else:
            raise ValueError(f"use_transformer type {use_transformer} not configured")

        self.full_embedding = TransformerEmbeddingModule(
            d_model=d_model, vocab_size=vocab_size, max_seq_length=max_seq_length, padding_idx=padding_idx,
            initilize_positional_with_sinusoids=initilize_positional_with_sinusoids
        )

        self.norm = nn.LayerNorm(d_model)
        self.to_token_map = nn.Linear(d_model, vocab_size, bias=False)

        # self.init_weights()


    def forward(self, src, tgt):
        """
        Forward pass of the Transformer model.

        Parameters
        ----------
        src : torch.Tensor of ints
            The source input tensor.
            Shape: [batch size, sequence length]
        tgt : torch.Tensor of ints
            The target input tensor.
            Shape: [batch size, sequence length]

        Returns
        -------
        torch.Tensor
            The output tensor of the Transformer model.
            Shape: [batch size, sequence length, vocabulary size]
        """
        tgt_mask = self.generate_square_subsequent_mask( tgt.size(1) ).detach().to(tgt.device)
        x = self.full_embedding(src)
        y = self.full_embedding(tgt)
        src_padding_mask = (src == self.padding_idx)
        tgt_padding_mask = (tgt == self.padding_idx)
        out = self.main_transformer(src=x, tgt=y,
                                    tgt_mask=tgt_mask,
                                    src_key_padding_mask=src_padding_mask,
                                    tgt_key_padding_mask=tgt_padding_mask,
                                    memory_key_padding_mask=src_padding_mask
        )
        out = self.norm(out)
        out = self.to_token_map(out)
        # return F.log_softmax(y, dim=-1) # -> for use with NLLLoss
        return out


    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square subsequent mask for use in the transformer's decoder.
        This is just a wrapper!
        Assumes that the underlying transformer implements the actual method.

        Parameters
        ----------
        sz : int
            The size of the mask.

        Returns
        -------
        torch.Tensor
            The square subsequent mask.
            Shape: [size, size]
        """
        return self.main_transformer.generate_square_subsequent_mask(sz)



##################################################################################
### Transformer from Scratch                                                   ###
##################################################################################

class MultiAttentionHeads_test(nn.Module):

    def __init__(self, embedding_dim, nr_attention_heads):
        """Wrapper around the pytorch function. To be used for debugging to identify whether a bug is in the Attention or elsewhere"""
        super(MultiAttentionHeads_test, self).__init__()

        self.embedding_dim = embedding_dim
        self.nr_attention_heads = nr_attention_heads
        self.attn = torch.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=nr_attention_heads, batch_first=True)


    def forward(self, x, y=None, mask=None, key_padding_mask=None):
        """for decoder with memory y can be used"""
        if y is not None:
            return self.attn(query=x, key=y, value=y, attn_mask=mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        else:
            return self.attn(query=x, key=x, value=x, attn_mask=mask, key_padding_mask=key_padding_mask, need_weights=False)[0]



class MultiAttentionHeads(nn.Module):

    def __init__(self, embedding_dim, nr_attention_heads):
        """This was my first attempt to implement multi-headed-attention. I tried to loop over every attention head.
        Something is not properly aligned with the concatentation and the masking of tokens (probably some broadcasting going awry).
        Please use the newer and more efficient NewMultiAttentionHeads below!
        """
        super(MultiAttentionHeads, self).__init__()

        self.embedding_dim = embedding_dim
        self.nr_attention_heads = nr_attention_heads


        self.attention_parts = ['q', 'k', 'v'] # list for easier iteration
        self.attention_dict = nn.ModuleList() # no longer a dict, but keeping old name

        assert (embedding_dim % nr_attention_heads) == 0, "The embedding_dim must be divisible by nr_attention_heads, because it is split"
        self.per_head_dim = embedding_dim // nr_attention_heads
        self.sqrt_per_head_dim  = math.sqrt(self.per_head_dim )


        for head_idx in range(nr_attention_heads):
            self.attention_dict.append(torch.nn.ModuleDict())
            for part in self.attention_parts:
                incl_bias=True if part=='q' else False
                # I saw in the reference implementation that pytorch only allows bias for the query by default, so I do the same here
                self.attention_dict[head_idx][part] = nn.Linear(embedding_dim, self.per_head_dim, bias=incl_bias)

        self.head_reducer = nn.Linear( embedding_dim, embedding_dim )
        # The name 'head_reducer' was based on my misunderstanding. I originally thought that embedding_dim is the dim per head,
        # but I couldn't match parameter counts with this.


    def forward(self, x, y=None, mask=None, key_padding_mask=None):
        """for decoder with memory y can be used"""
        head_parts_collector = []


        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1)#.unsqueeze(1)
            # above will make the shape of key_padding_mask (batch_size, 1, 1, seq_len),
            # which can be broadcasted to match the shape of mm by the masked fill.

        for head_idx, curr_head in enumerate(self.attention_dict):
            # for decoder query=x, key='from encoder/memory', value='from encoder/memory'
            q = self.attention_dict[head_idx]['q'](x)
            if y is not None:
                k = self.attention_dict[head_idx]['k'](y)
                v = self.attention_dict[head_idx]['v'](y)
            else:
                k = self.attention_dict[head_idx]['k'](x)
                v = self.attention_dict[head_idx]['v'](x)

            mm = (torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_per_head_dim)

            if mask is not None:
                mm = mm + mask
            if key_padding_mask is not None:
                mm = mm.masked_fill(key_padding_mask, float('-inf'))

            q_k = nn.functional.softmax(mm, dim=-1)

            head_res = torch.matmul(q_k.transpose(-2, -1), v)
            head_parts_collector.append(head_res)

        heads_out = torch.cat(head_parts_collector, dim=-1)

        # print(f"heads_out shape: {heads_out.shape}")
        #print(f"heads_out shape: {heads_out.shape}")
        return self.head_reducer(heads_out)




class NewMultiAttentionHeads(nn.Module):
    def __init__(self, embedding_dim, nr_attention_heads):
        """
        A multi-head attention module, implementing the scaled dot-product attention mechanism.
        This is the properly working and efficient implementation.

        Parameters
        ----------
        embedding_dim : int
            The dimension of the input embeddings.
        nr_attention_heads : int
            The number of attention heads.

        Attributes
        ----------
        q_lin : torch.nn.Linear
            The linear layer for the query matrix (of all heads).
        k_lin : torch.nn.Linear
            The linear layer for the key matrix (of all heads).
        v_lin : torch.nn.Linear
            The linear layer for the value matrix (of all heads).
        head_reducer : torch.nn.Linear
            The linear layer to reduce the multi-head attention output to the original embedding dimension.
        """
        super(NewMultiAttentionHeads, self).__init__()

        self.embedding_dim = embedding_dim
        self.nr_attention_heads = nr_attention_heads
        self.per_head_dim = embedding_dim // nr_attention_heads
        self.sqrt_per_head_dim = math.sqrt(self.per_head_dim)

        assert (embedding_dim % nr_attention_heads) == 0, "The embedding_dim must be divisible by nr_attention_heads, because it is split"

        self.q_lin = nn.Linear(embedding_dim, embedding_dim)
        self.k_lin = nn.Linear(embedding_dim, embedding_dim)
        self.v_lin = nn.Linear(embedding_dim, embedding_dim)

        self.head_reducer = nn.Linear(embedding_dim, embedding_dim)


    def split_heads(self, x, batch_size):
        """
        Split the input tensor into attention heads.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            Shape: [batch size, sequence length, embedding dim]
        batch_size : int
            The size of the batch.

        Returns
        -------
        torch.Tensor
            The reshaped input tensor.
            Shape: [batch size, nr_attention_heads, sequence length, per_head_dim]
        """
        x = x.view(batch_size, -1, self.nr_attention_heads, self.per_head_dim)
        return x.permute(0, 2, 1, 3)


    def forward(self, x, y=None, mask=None, key_padding_mask=None):
        """
        Forward pass of the multi-head attention module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            Shape: [batch size, sequence length, embedding dim]
        y : torch.Tensor, optional
            The optional input tensor for cross-attention.
            Shape: [batch size, sequence length, embedding dim]
        mask : torch.Tensor, optional
            The mask to be applied to the attention scores.
            Shape: [batch size, 1, 1, sequence length]
        key_padding_mask : torch.Tensor, optional
            The mask for key padding.
            Shape: [batch size, sequence length]

        Returns
        -------
        torch.Tensor
            The output tensor of the multi-head attention.
            Shape: [batch size, sequence length, embedding dim]
        """
        batch_size = x.size(0)

        q = self.split_heads(self.q_lin(x), batch_size)
        k = self.split_heads(self.k_lin(y if y is not None else x), batch_size)
        v = self.split_heads(self.v_lin(y if y is not None else x), batch_size)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_per_head_dim

        if mask is not None:
            attn_scores = attn_scores + mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.embedding_dim)

        return self.head_reducer(context)



class TransformerEncoderBlock(nn.Module):

    def __init__(self, embedding_dim, nr_attention_heads, ffn_inner_size, dropout=0.1):
        """
        A Transformer encoder block, consisting of multi-head attention and feed-forward layers.

        Parameters
        ----------
        embedding_dim : int
            The dimension of the input embeddings.
        nr_attention_heads : int
            The number of attention heads.
        ffn_inner_size : int
            The inner size of the feed-forward network.
        dropout : float, optional, default=0.1
            The dropout rate.

        Attributes
        ----------
        attention : NewMultiAttentionHeads
            The multi-head attention layer.
        norm1, norm2 : torch.nn.LayerNorm
            Layer normalization layers.
        ffn_1, ffn_2 : torch.nn.Linear
            Linear layers for the feed-forward network.
        dropout : torch.nn.Dropout
            The dropout layer.
        activation : torch.nn.GELU
            The activation function for the feed-forward network.
        """
        super(TransformerEncoderBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.nr_attention_heads = nr_attention_heads

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # For the FNN layer
        self.ffn_1 = nn.Linear(embedding_dim, ffn_inner_size)
        self.ffn_2 = nn.Linear(ffn_inner_size, embedding_dim)

        self.attention = NewMultiAttentionHeads(
            embedding_dim=embedding_dim,
            nr_attention_heads=nr_attention_heads
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)


    def forward(self, x, key_padding_mask=None):
        """
        Forward pass of the Transformer encoder block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            Shape: [batch size, sequence length, embedding dim]
        key_padding_mask : torch.Tensor, optional
            The mask for key padding.
            Shape: [batch size, sequence length]

        Returns
        -------
        torch.Tensor
            The output tensor of the Transformer encoder block.
            Shape: [batch size, sequence length, embedding dim]
        """
        # Apply Multiheaded self-attention
        x_new = self.norm1(x) # Pre-LN version
        x_new = self.attention(x_new, key_padding_mask=key_padding_mask)

        # Add Residual and Normalize (original version for Post-LN)
        x = x_new + x

        # Apply FFN layers
        x_new = self.norm2(x) # Pre-LN version
        x_new = self.ffn_1(x_new)
        x_new = self.activation(x_new)
        x_new = self.dropout(x_new)
        x_new = self.ffn_2(x_new)

        # Add Residual and Normalize (original version Post-LN)
        x = x_new + x

        return x



class TransformerDecoderBlock(nn.Module):

    def __init__(self, embedding_dim, nr_attention_heads, ffn_inner_size, dropout=0.1):
        """
        A Transformer decoder block, consisting of multi-head attention, memory attention, and feed-forward layers.

        Parameters
        ----------
        embedding_dim : int
            The dimension of the input embeddings.
        nr_attention_heads : int
            The number of attention heads.
        ffn_inner_size : int
            The inner size of the feed-forward network.
        dropout : float, optional, default=0.1
            The dropout rate.

        Attributes
        ----------
        attention : NewMultiAttentionHeads
            The multi-head attention layer for self-attention.
        attention_memory : NewMultiAttentionHeads
            The multi-head attention layer for memory attention.
        norm1, norm2, norm3 : torch.nn.LayerNorm
            Layer normalization layers.
        ffn_1, ffn_2 : torch.nn.Linear
            Linear layers for the feed-forward network.
        dropout : torch.nn.Dropout
            The dropout layer.
        activation : torch.nn.GELU
            The activation function for the feed-forward network.
        """
        super(TransformerDecoderBlock, self).__init__()


        self.embedding_dim = embedding_dim
        self.nr_attention_heads = nr_attention_heads


        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # For the FNN layer
        self.ffn_1 = nn.Linear(embedding_dim, ffn_inner_size)
        self.ffn_2 = nn.Linear(ffn_inner_size, embedding_dim)

        self.attention = NewMultiAttentionHeads(
            embedding_dim=embedding_dim,
            nr_attention_heads=nr_attention_heads
        )

        self.attention_memory = NewMultiAttentionHeads(
            embedding_dim=embedding_dim,
            nr_attention_heads=nr_attention_heads
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)


    def forward(self, x, encoder_memory, mask, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass of the Transformer decoder block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            Shape: [batch size, sequence length, embedding dim]
        encoder_memory : torch.Tensor
            The memory tensor from the encoder (i.e. output of the encoder).
            Shape: [batch size, sequence length, embedding dim]
        mask : torch.Tensor
            The mask to be applied to the attention scores.
            Shape: [batch size, 1, 1, sequence length]
        src_key_padding_mask : torch.Tensor, optional
            The mask for source key padding.
            Shape: [batch size, sequence length]
        tgt_key_padding_mask : torch.Tensor, optional
            The mask for target key
            Shape: [batch size, sequence length]

        Returns
        -------
        torch.Tensor
            The output tensor of the Transformer encoder block.
            Shape: [batch size, sequence length, embedding dim]
        """
        # Apply Multiheaded self-attention
        x_new = self.norm1(x)
        x_new = self.attention(x_new, mask=mask, key_padding_mask=tgt_key_padding_mask)

        # Add Residual and Normalize
        x = x_new + x

        ## print(f"encoder memory shape (in Decoder): {encoder_memory.shape}")

        # Apply Multiheaded memory-enconder-attention
        x_new = self.norm2(x)
        x_new = self.attention_memory(x_new, encoder_memory, key_padding_mask=src_key_padding_mask)

        # Add Residual and Normalize
        x = x_new + x


        # Apply FFN layers
        x_new = self.norm3(x)
        x_new = self.ffn_1(x_new)
        x_new = self.activation(x_new)
        x_new = self.dropout(x_new)
        x_new = self.ffn_2(x_new)

        # Add Residual and Normalize
        x = x_new + x

        return x


class TransformerBlockFromScratch(nn.Module):

    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        A custom Transformer block, consisting of a stack of encoder and decoder layers.

        Parameters
        ----------
        num_encoder_layers : int
            The number of encoder layers in the Transformer block.
        num_decoder_layers : int
            The number of decoder layers in the Transformer block.
        d_model : int
            The dimension of the input embeddings.
        nhead : int
            The number of attention heads.
        dim_feedforward : int
            The inner size of the feed-forward networks in the encoder and decoder layers.
        dropout : float, optional, default=0.1
            The dropout rate.

        Attributes
        ----------
        encoder : torch.nn.ModuleList
            A list of TransformerEncoderBlock layers.
        decoder : torch.nn.ModuleList
            A list of TransformerDecoderBlock layers.
        """
        super(TransformerBlockFromScratch, self).__init__()

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim=d_model, nr_attention_heads=nhead, ffn_inner_size=dim_feedforward)
            for i in range(num_encoder_layers)
        ])

        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(embedding_dim=d_model, nr_attention_heads=nhead, ffn_inner_size=dim_feedforward)
            for i in range(num_decoder_layers)
        ])

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the custom Transformer block.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor for the encoder.
            Shape: [batch size, sequence length, embedding dim]
        tgt : torch.Tensor
            The input tensor for the decoder.
            Shape: [batch size, sequence length, embedding dim]
        tgt_mask : torch.Tensor, optional
            The mask to be applied to the decoder's attention scores.
            Shape: [batch size, 1, 1, sequence length]
        src_key_latitude, tgt_key_padding_mask : torch.Tensor, optional
            The mask for source/target key padding.
            Shape: [batch size, sequence length]
        memory_key_padding_mask : torch.Tensor, optional
            The mask for memory key padding, currently unused.
            Shape: [batch size, sequence length]

        Returns
        -------
        torch.Tensor
            The output tensor of the custom Transformer block.
            Shape: [batch size, sequence length, embedding dim]
        """
        for idx, enc in enumerate(self.encoder):
            src = enc(src, key_padding_mask=src_key_padding_mask)
        for idx, dec in enumerate(self.decoder):
            tgt = dec(tgt, encoder_memory=src, mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return tgt

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on the diagonal.

        Parameters
        ----------
        sz : int
            The size of the square matrix.

        Returns
        -------
        torch.Tensor
            An upper-triangular matrix of -inf, with zeros on the diagonal.
            Shape: [sz, sz]
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
