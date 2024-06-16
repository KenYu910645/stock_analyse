import torch.nn as nn

NUM_HIDEN_FEATURE = 256 # 64

class Minion(nn.Module):
    '''
    Minion moodel
    '''
    def __init__(self, num_in_feature):
        super(Minion, self).__init__()

        # Define layers
        layers = [
            nn.Linear(num_in_feature, NUM_HIDEN_FEATURE),
            nn.ReLU(),
            nn.Linear(NUM_HIDEN_FEATURE, NUM_HIDEN_FEATURE),
            nn.ReLU(),
            nn.Linear(NUM_HIDEN_FEATURE, NUM_HIDEN_FEATURE),
            nn.ReLU(),
            nn.Linear(NUM_HIDEN_FEATURE, 4)  # Output 4 values
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
