import torch.nn as nn
from config import cfg

class FCN(nn.Module):
    '''
    Fully connected network model.
    '''
    def __init__(self, num_in_feature):
        super(FCN, self).__init__()

        # Input layers
        layers = [
            nn.Linear(num_in_feature, cfg.num_hidden_feature),
            nn.ReLU(),
        ]

        # Hidden layers
        for _ in range(cfg.num_hidden_layer):
            layers += [
                nn.Linear(cfg.num_hidden_feature, cfg.num_hidden_feature),
                nn.ReLU(),
            ]

        # Output layers
        layers += [
            nn.Linear(cfg.num_hidden_feature, 4)  # Output 4 values
        ]

        # Use nn.Sequential to create the network
        self.network = nn.Sequential(*layers)

        # Apply sigmoid to the confidence value
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.network(x)
        # x[:, -1] = self.sigmoid(x[:, -1])  # Apply sigmoid to the confidence value
        return x
