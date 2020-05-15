import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import math

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size = 1, stride = 1, padding = 0)
        # self.pool = nn.MaxPool2d(3,1)
        nn.init.orthogonal_(self.conv1.weight, numpy.sqrt(2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 1, stride = 1, padding = 0)
        nn.init.orthogonal_(self.conv2.weight, numpy.sqrt(2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=  1, stride = 1, padding = 0)
        nn.init.orthogonal_(self.conv3.weight, numpy.sqrt(2))

        self.fc1 = nn.Linear(in_features = 28 * 28 * 64, out_features = 512)
        nn.init.orthogonal_(self.fc1.weight, numpy.sqrt(2))


        #The following two fc layers aren't connected, they are just two separate output branches 

        # Pi logits needed for train, more explained on the doc provided
        # A fully connected layer to get logits for π
        # Linear values that can be transformed into probabilities for actions
        self.pi_logits = nn.Linear(in_features = 512, out_features = 3)
        nn.init.orthogonal_(self.pi_logits.weight, numpy.sqrt(.01))


    def forward(self, obs: numpy.ndarray):
        h: torch.Tensor

        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape((-1, 28 * 28 * 64))
        h = F.relu(self.fc1(h))

        action = torch.distributions.categorical.Categorical(logits = self.pi_logits(h))

        return action

"""

Define NN
50000 = net.conv1.weight.data.shape() + ....

net = Net()
env = gym.make(...) OR Wrrappers.CroppedImage('binary-narrow')

class Chromosome:
    def __init__(self):
        self._genes = #50000 numpy array
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