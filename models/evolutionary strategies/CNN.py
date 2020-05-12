import torch.nn as nn
import torch.nn.functional as F
import geneticAlgorithmScript
import numpy
import math

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(3,1)
        self.conv2 = nn.Conv2d(4, 64, kernel_size=3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(4, 64, kernel_size=3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def backward(self, x, totAccumulatedReward):
        weightsPre = self.fc3.weight.data
        weightsInput = weightsPre.numpy()

        minW = math.inf
        maxW = -math.inf

        for r in range(0, weightsInput.shape[0]):
            for c in range(0, weightsInput.shape[1]):
                if weightsInput[r,c] < minW:
                    minW = weightsInput[r,c]
                if weightsInput[r,c] > maxW:
                    maxW = weightsInput[r,c]

        
        x = geneticAlgorithmScript.geneticAlgorithmMain(weightsInput, weightsInput.shape[1], weightsInput.shape[1] * 1.5, minW, maxW, 40, weightsInput.shape[1] / 2, totAccumulatedReward)
        return x


net = Net()

## the output of our evolutionary strategies needs to be equal
## to the input to the convolutional neural network
