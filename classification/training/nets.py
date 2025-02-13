import torch.nn as nn
import torch

# simple MLP network
class SimpleMLP(nn.Module):
    def __init__(self, input_channels, hidden1, hidden2, output):
        super(SimpleMLP, self).__init__()
        
        self.linear1 = nn.Linear(input_channels, hidden1)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.activation2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden2, output)

    def forward(self, input):
        out = self.activation(self.linear1(input))
        out = self.activation2(self.linear2(out))
        out = self.output_layer(out)
        
        return out 

# LSTM network    
class LSTMNet(nn.Module):
    def __init__(self, input_channels, hidden_size, output, num_layers, dropout):
        super(LSTMNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True)
        
        # Output layer (fully connected layer after LSTM)
        self.fc = nn.Linear(hidden_size, output)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = self.dropout(output[:, -1, :])
        
        output = self.fc(output)  # (batch_size, output_size)
        
        return output
