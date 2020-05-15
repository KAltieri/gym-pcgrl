import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cpu")


class Net(nn.Module):
    def __init__(self, in_channels, map_size):
        super().__init__()
        self.map_size = map_size
        #print(in_channels);
        #print(in_channels);
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1, stride=1, padding=0)
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))

        self.fc1 = nn.Linear(in_features = map_size * map_size * 64, out_features = 512)
        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))

        #self.lin = nn.Linear(in_features=map_size * map_size * 64,out_features=512)
        #nn.init.orthogonal_(self.lin.weight, np.sqrt(2))

        # Pi logits needed for train, more explained on the doc provided
        # A fully connected layer to get logits for π
        # Linear values that can be transformed into probabilities for actions
        self.pi_logits = nn.Linear(in_features = 512, out_features = 3)
        nn.init.orthogonal_(self.pi_logits.weight, np.sqrt(.01))


    def forward(self, obs):
        h: torch.Tensor

        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape((-1, self.map_size * self.map_size * 64))

        h = F.relu(self.fc1(h))
        action = torch.distributions.categorical.Categorical(logits = self.pi_logits(h))

        #print(action)
        #print()
        #print(action.sample(torch.Size([1,3])))
        #print()

        actionGet = action.sample()
        a = actionGet.cpu().data.numpy()

        return a[0]


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    #print(obs)
    obs = np.reshape(obs, (28, 28, 1, 1))
    obs = np.swapaxes(obs, 1, 3)
    # print("after first",obs.shape)
    obs = np.swapaxes(obs, 3, 2)
    # float32
    return torch.tensor(obs, dtype= torch.double, device=device)

