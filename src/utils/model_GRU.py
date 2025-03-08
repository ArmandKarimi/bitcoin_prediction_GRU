import torch
import torch.nn as nn

class BitcoinGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout_rate=0.2):
        """
        A simple multi-layer GRU for time-series regression.

        :param input_size:   Number of input features at each timestep
        :param hidden_size:  Dimensionality of the GRU hidden state
        :param num_layers:   Number of stacked GRU layers
        :param dropout_rate: Dropout probability between GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU: 
        # - batch_first=True means input shape is (batch_size, sequence_length, input_size)
        # - dropout applies *between* layers (for num_layers > 1)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        # Final fully connected layer to produce a single value (regression)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the GRU model.

        :param x: Input tensor of shape (batch_size, seq_length, input_size)
        :return:  Prediction for the last timestep (batch_size, 1)
        """

        # out shape: (batch_size, seq_length, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        out, hidden = self.gru(x)

        # We only want the last timestep's output for next-day prediction
        # out[:, -1, :] => shape (batch_size, hidden_size)
        out = out[:, -1, :]

        # Pass through the fully connected layer
        out = self.fc(out)  # shape (batch_size, 1)
        return out
