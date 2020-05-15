import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class Net(nn.Module):
    def __init__(self, in_channels, map_size, out_length):
        super().__init__()
        self.map_size = map_size
        #print(in_channels);
        print(in_channels);
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

        action = torch.distributions.categorical.Categorical(logits = self.pi_logits(h))

        return action

    def gene_size(self):
        genes = np.zeros(list(torch.flatten(self.conv1.weight).size())[0] + list(torch.flatten(self.conv1.bias).size())[0]
            + list(torch.flatten(self.conv2.weight).size())[0] + list(torch.flatten(self.conv2.bias).size())[0]
            + list(torch.flatten(self.conv3.weight).size())[0] + list(torch.flatten(self.conv3.bias).size())[0]
            + list(torch.flatten(self.fc1.weight).size())[0] + list(torch.flatten(self.fc1.bias).size())[0]
            + list(torch.flatten(self.pi_logits.weight).size())[0] + list(torch.flatten(self.pi_logits.bias).size())[0])
        return genes

def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    #print(obs)
    #obs = np.swapaxes(obs, 1, 3)
    # print("after first",obs.shape)
    #obs = np.swapaxes(obs, 3, 2)
    # float32
    return torch.tensor(obs, dtype= torch.float32, device=device)

"""

Define NN
50000 = net.conv1.weight.data.shape() + ....

net = Net()
env = gym.make(...) OR Wrrappers.CroppedImage('binary-narrow')

class Chromosome:
    def __init__(self):
        self._genes = #50000 np array
        self._fitness = 0

    def copy(self):
        c = Chromosome()
        c._genes = self._genes.copy()
        return c

    def mutation(self):
        self.genes += np.guassian(size, mean=0, std=0.1)

    def fitness(self, env, net, episode_num):
        #initialize network based on the chromosome
        net.conv1.weight.data = self._genes[0:400]
        net.conv2.weight.data = self._genes[400:800]
        ..
        ..
        ..
        #calculate fitness
        totalReward = 0
        for i in range(episode_num):
            obs = env.reset()
            done = false
            while not done:
                action = net.forward(obs)
                obs, reward, done, _ = env.step(action)
                totalReward += reward
        self._fitness = totalReward / (numberEpisode * 1.0)
        return self._fitness



class GA:
    def __init__(self, popSize, mu, lamda):
        self._pop = []
        for i in range(popSize):
            self._pop.append(Chromosome())

        self.mu = mu
        self.lamda = lamda

    def advance(self):
        for c in self._pop:
            #net defined outside
            c.fitness(env, net, 20)

        sort(self._pop, lamda c: c._fitness, reverse = True)
        for i in range(self.mu):
            c = self._pop[i].copy
            c = c.mutate()
            self._pop[self.lamda + 1] = c

    def run(self, generations):
        for i in range(generations):
            self.advance()


ga = GA(100, 50, 50)
ga,run(1000)


"""
