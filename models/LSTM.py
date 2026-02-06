from torch import nn


class LSTM_two_layers(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        # Define the LSTM layer: processes sequential data
        self.lstm = nn.LSTM(input_size = input_size, # size of the vector for temporal instance
                            hidden_size = hidden_size, # size of the internal state
                            num_layers = 2, # number of stacked LSTM layers
                            batch_first=True,
                            bidirectional=False,
                            dropout = dropout) # input will have the form (batch, seq_len, features)
        
        # Final dense layer: maps hidden state output to predictions
        self.fc = nn.Linear(hidden_size, output_size) 

    # Define how the information flows through the model
    def forward(self, x): 
        # x tensor shape: (batch, seq_len, input_size)
        # out: all hidden states, h_n: last hidden state, c_n: last cell state
        out, (h_n, c_n) = self.lstm(x) 
        
        # Extract the last hidden state from the sequence
        last_hidden = out[:, -1, :]
        
        # Apply the linear layer to the last time step
        y_hat = self.fc(last_hidden) 
        return y_hat