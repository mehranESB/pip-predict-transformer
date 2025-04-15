import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EmbeddingLayer(nn.Module):
    """
    Embedding Layer for preparing input for transformer encoders with blocks and residual connections.

    Args:
        in_channels (int): Number of input channels (features in the sequence).
        d_model (int): Dimension of the embedding space (output features).
        num_blocks (int): Number of blocks of embedding layers.
        num_layers (int): Number of fully connected layers in each block.
        num_groups (int): Number of groups for group normalization.
        act_fun (callable): Activation function to use after each layer.
    """

    def __init__(
        self,
        in_channels,
        d_model,
        num_blocks=1,
        num_layers=1,
        num_groups=1,
        act_fun=F.relu,
    ):
        super(EmbeddingLayer, self).__init__()

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.act_fun = act_fun

        # Define blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            layers = []
            for _ in range(num_layers):
                # Fully connected layer
                fc = nn.Linear(in_channels, d_model)
                # Group Normalization: normalizes channels within groups for each sequence
                gn = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
                layers.append(nn.Sequential(fc, gn))
                # Update in_channels to match the d_model for subsequent layers
                in_channels = d_model
            self.blocks.append(nn.ModuleList(layers))

        # Residual normalization layer after each block (except the first block)
        self.block_norms = nn.ModuleList(
            [
                nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
                for _ in range(num_blocks - 1)
            ]
        )

    def forward(self, x):
        """
        Forward pass for the embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, in_channels).

        Returns:
            torch.Tensor: Transformed tensor of shape (batch, seq_len, d_model).
        """
        for block_idx, block in enumerate(self.blocks):
            if block_idx == 0:
                # First block: no residual connection, just pass through
                for layer in block:
                    fc, gn = layer

                    # Apply linear transformation
                    x = fc(x)

                    # Permute to (batch, d_model, seq_len) for GroupNorm
                    x = x.permute(0, 2, 1)
                    x = gn(x)  # GroupNorm operates on channels

                    # Permute back and apply activation
                    x = x.permute(0, 2, 1)
                    x = self.act_fun(x)
            else:
                # Subsequent blocks: with residual connections
                residual = x  # Save input for residual connection
                for layer in block:
                    fc, gn = layer

                    # Apply linear transformation
                    x = fc(x)

                    # Permute to (batch, d_model, seq_len) for GroupNorm
                    x = x.permute(0, 2, 1)
                    x = gn(x)

                    # Permute back and apply activation
                    x = x.permute(0, 2, 1)
                    x = self.act_fun(x)

                # Apply residual connection and normalization after block
                x += residual  # Residual connection
                x = x.permute(0, 2, 1)  # Permute for GroupNorm
                # Norm after residual (block_idx - 1 because first block has no norm)
                x = self.block_norms[block_idx - 1](x)
                x = x.permute(0, 2, 1)  # Permute back

        return x


class PositionalEncoding(nn.Module):
    """
    Add sinusoidal positional encoding to the input tensor.

    Args:
        d_model (int): Dimension of the model (must match the last dimension of the input).
        max_len (int): Maximum length of the sequence to support.
    """

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        # Create a matrix to store positional encodings of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Create a position tensor (0, 1, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the div_term for alternating sine and cosine
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it is not updated during backpropagation
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Input tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        # Add positional encoding up to the sequence length
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of N TransformerEncoderLayers.

    Args:
        d_model (int): Dimension of the embedding space (input and output features).
        nhead (int): Number of attention heads in each encoder layer.
        num_layers (int): Number of encoder layers in the stack.
        dim_feedforward (int): Dimension of the feedforward network in each encoder layer. Default is 2048.
        dropout (float): Dropout rate for the encoder layers. Default is 0.1.
        activation (callable): Activation function to use in the encoder layers. Default is F.relu.
    """

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
    ):
        super(TransformerEncoder, self).__init__()

        # Define a single TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Ensures input is of shape (batch, seq_len, d_model)
        )

        # Stack multiple layers to form the complete Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        """
        Forward pass for the Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch, seq_len, d_model).
        """
        # Pass input through the stacked encoder layers
        output = self.encoder(x)
        return output


class OutputProjectionLayer(nn.Module):
    """
    A layer to project Transformer encoder outputs to (batch, seq_len, 1).

    Args:
        d_model (int): Dimension of the input features from the Transformer encoder.
    """

    def __init__(self, d_model, LTR):
        super(OutputProjectionLayer, self).__init__()
        # Linear layer to map d_model to 1
        self.fc = nn.Linear(d_model, 1)
        # if model need to learn to sort
        self.ltr = LTR

    def forward(self, x):
        """
        Forward pass for the OutputProjectionLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Projected tensor of shape (batch, seq_len) with ReLU activation applied.
        """
        # Apply linear transformation
        x = self.fc(x)
        # Apply activation
        if self.self.ltr:
            x = x.squeeze(-1)  # not necessary for activation
        else:
            x = F.tanh(x).squeeze(-1)
        return x
