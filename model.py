import torch.nn as nn

class Minion(nn.Module):
    def __init__(self, num_in_feature):
        super(Minion, self).__init__()
        self.fc1 = nn.Linear(num_in_feature, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # x[:, -1] = self.sigmoid(x[:, -1])  # Apply sigmoid to the confidence value
        return x
