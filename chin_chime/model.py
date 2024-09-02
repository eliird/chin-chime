import torch.nn as nn
import torch

class MLPLandmark(nn.Module):
    def __init__(self, inp_landmarks: int, out_dim: int, layers: list, dim_per_landmark: int=3):
        super().__init__()
        
        self.inp_dim = inp_landmarks * dim_per_landmark
        
        self.mlp = nn.ModuleList()
        
        self.mlp.append(nn.Linear(self.inp_dim, layers[0]))
        
        for i in range(1, len(layers)):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(layers[i-1], layers[i]),
                    nn.ReLU(),
                    nn.LayerNorm(layers[i])
                )
            )
        
        self.mlp.append(nn.Linear(layers[-1], out_dim))
        
    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], self.inp_dim)
        
        for layer in self.mlp:
            x = layer(x)
        
        return x
    

class ANN(nn.Module):
    def __init__(self, in_feats: int, num_labels: int, hidden_layers: list):
        super(ANN, self).__init__()

        # Define the first layer (input layer)
        self.input_layer = nn.Linear(in_feats, hidden_layers[0])

        # Define the hidden layers dynamically
        self.mlp = nn.ModuleList()

        for i in range(1, len(hidden_layers)):
            self.mlp.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.mlp.append(nn.ReLU())

        # Define the output layer
        self.output_layer = nn.Linear(hidden_layers[-1], num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through input layer
        x = self.input_layer(x)
        x = nn.ReLU()(x)

        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Forward pass through output layer and activation
        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x
