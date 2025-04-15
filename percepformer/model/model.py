from .layer import (
    EmbeddingLayer,
    PositionalEncoding,
    TransformerEncoder,
    OutputProjectionLayer,
)
import torch.nn as nn
import torch.nn.functional as F

# Map string to function
act_funs = {
    "relu": F.relu,
    "gelu": F.gelu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "softmax": F.softmax,
}


class PercepFormer(nn.Module):
    """
    PercepFormer: A transformer-based architecture for processing sequential data.

    Args:
        embedding (EmbeddingLayer): Instance of the EmbeddingLayer for input data.
        positional_encoding (PositionalEncoding): Instance of the PositionalEncoding layer.
        transformer_encoder (nn.TransformerEncoder): Instance of the TransformerEncoder.
        output_projection (OutputProjectionLayer): Instance of the OutputProjectionLayer.
    """

    def __init__(
        self,
        embedding: EmbeddingLayer,
        positional_encoding: PositionalEncoding,
        transformer_encoder: TransformerEncoder,
        output_projection: OutputProjectionLayer,
    ):
        super(PercepFormer, self).__init__()

        self.embedding = embedding
        self.positional_encoding = positional_encoding
        self.transformer_encoder = transformer_encoder
        self.output_projection = output_projection

    def forward(self, x):
        """
        Forward pass for PercepFormer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, in_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, 2).
        """
        # Step 1: Embedding layer
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # Step 2: Positional Encoding
        x = self.positional_encoding(x)  # (batch, seq_len, d_model)

        # Step 3: Transformer Encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Step 4: Output Projection
        x = self.output_projection(x)  # (batch, seq_len, 1)

        return x


def create_model(**kwargs):
    """
    Function to create an instance of the PercepFormer model with the provided arguments.

    Args:
        **kwargs: Arbitrary keyword arguments used to define the model components.
                  Expected arguments:
                  - in_channels (int): Number of input channels (features in the sequence).
                  - d_model (int): Dimension of the embedding space.
                  - num_layers (int): Number of fully connected layers in the EmbeddingLayer.
                  - num_groups (int): Number of groups for Group Normalization in the EmbeddingLayer.
                  - embed_act_fun (callable):  Activation function to use after each layer in the EmbeddingLayer.
                  - act_fun (callable): Activation function to use after each layer in the Transformers.
                  - nhead (int): Number of attention heads in the Transformer encoder.
                  - num_encoder_layers (int): Number of Transformer encoder layers.
                  - dim_feedforward (int): Feedforward dimension in the Transformer encoder.
                  - dropout (float): Dropout rate in the Transformer encoder.

    Returns:
        PercepFormer: An instance of the PercepFormer model.
    """

    # Extract arguments from kwargs (with defaults where necessary)
    in_channels = kwargs.get("in_channels", 5)  # Default 5
    d_model = kwargs.get("d_model", 32)  # Default 32
    num_layers = kwargs.get("num_layers", 2)  # Default 2
    num_groups = kwargs.get("num_groups", 4)  # Default 4
    embed_act_fun = act_funs[kwargs.get("embed_act_fun", "tanh")]  # Default Tanh
    act_fun = act_funs[kwargs.get("act_fun", "relu")]  # Default ReLU
    nhead = kwargs.get("nhead", 4)  # Default 4
    num_encoder_layers = kwargs.get("num_encoder_layers", 4)  # Default 4
    dim_feedforward = kwargs.get("dim_feedforward", 128)  # Default 128
    dropout = kwargs.get("dropout", 0.1)  # Default 0.1
    ltr = kwargs.get("learn_to_sort", False)  # Default False

    # Initialize the layers based on provided parameters
    embedding = EmbeddingLayer(
        in_channels=in_channels,
        d_model=d_model,
        num_layers=num_layers,
        num_groups=num_groups,
        act_fun=embed_act_fun,
    )

    positional_encoding = PositionalEncoding(d_model=d_model)

    transformer_encoder = TransformerEncoder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=act_fun,
    )

    output_projection = OutputProjectionLayer(d_model=d_model, LTR=ltr)

    # Create and return the PercepFormer model
    model = PercepFormer(
        embedding=embedding,
        positional_encoding=positional_encoding,
        transformer_encoder=transformer_encoder,
        output_projection=output_projection,
    )

    return model
