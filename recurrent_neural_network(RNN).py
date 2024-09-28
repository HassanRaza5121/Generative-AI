import torch
import torch.nn as nn

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        
        # RNN layer: input_size is the dimensionality of input, hidden_size is the number of features in the hidden state
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer (fully connected layer)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        
        # Forward propagate through RNN
        out, hn = self.rnn(x, h0)  # out: batch_size, seq_length, hidden_size
        
        # Pass the output of the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])  # Taking the output at the last time step
        return out

# Define hyperparameters
input_size = 5  # Number of input features (e.g., word embedding size)
hidden_size = 10  # Number of hidden state features
output_size = 1  # Number of output features (e.g., for regression or binary classification)
num_layers = 1  # Number of RNN layers

# Instantiate the model, define loss function and optimizer
model = SimpleRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example input (batch_size, seq_length, input_size)
# e.g., batch of 3 sequences, each of length 4, and each element of the sequence is a vector of size input_size
input_data = torch.randn(3, 4, input_size)

# Forward pass
output = model(input_data)
print("Model output shape:", output.shape)

# Example target (for training purposes)
target = torch.randn(3, output_size)

# Compute loss
loss = criterion(output, target)
print("Loss:", loss.item())

# Backward pass and optimization
loss.backward()
optimizer.step()
