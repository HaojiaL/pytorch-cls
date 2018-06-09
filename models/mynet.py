import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 7)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv1_1 = nn.Conv2d(6, 16, 3, 2, 1)
        self.conv2_1 = nn.Conv2d(32, 32, 3, 2, 1)
        self.fc1 = nn.Linear(32 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv1_1(out))
        # out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv2_1(out))
        # out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out