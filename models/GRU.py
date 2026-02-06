import torch
import torch.nn as nn

class GRU_two_layers(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        # Define the GRU layer: processes sequential data with gated recurrent units
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=False, 
            dropout=dropout
        )
        
        # Final dense layer: maps the last hidden state to the prediction output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Passes the complete sequence through the GRU layer
        # out: hidden states for all time steps, h_n: last hidden state for all layers
        out, h_n = self.gru(x)
        
        # Extract the hidden state from the last time step which contains 
        # information from the entire sequence
        last_hidden = out[:, -1, :]
        
        # Apply the linear layer to get the final predictions
        y_hat = self.fc(last_hidden)
        return y_hat