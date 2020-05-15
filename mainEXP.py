import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gym
from gym_pcgrl import wrappers

import geneticAlgorithmScript
import CNN

class Chromosome:
    def __init__(self, net):
        self.kwargs = {
            'change_percentage': 0.4,
            'verbose': True
        }
        # 28 possibly
        self.kwargs['cropped_size'] = 28
        self.crop_size = self.kwargs.get('cropped_size', 28)

        self._env = wrappers.CroppedImagePCGRLWrapper("binary-narrow-v0", self.crop_size, **self.kwargs)
        self._obs = np.zeros
        self._net = net
        self._fitness = 0

        self._genes = np.zeros(list(torch.flatten(self._net.conv1.weight).size())[0] + 
            + list(torch.flatten(self._net.conv1.bias).size())[0] + list(torch.flatten(self._net.conv2.weight).size())[0] 
            + list(torch.flatten(self._net.conv2.bias).size())[0]
            + list(torch.flatten(self._net.conv3.weight).size())[0] + list(torch.flatten(self._net.conv3.bias).size())[0]
            + list(torch.flatten(self._net.fc1.weight).size())[0] + list(torch.flatten(self._net.fc1.bias).size())[0]
            + list(torch.flatten(self._net.pi_logits.weight).size())[0] + list(torch.flatten(self._net.pi_logits.bias).size())[0])
        

    def copy(self):
        c = Chromosome()
        c._genes = self._genes.copy()
        return c

    def mutation(self):
        for idx in range(self._genes.shape[0]):
            random_value = np.random.normal(0, 0.1, 1)
            random_int = np.random.randint(0, 10)
            if random_int == 1:
                self._genes[idx] += random_value
        return

    def fitness(self, numberEpisodes):
        #Conv1 and Bias1, and so on and so forth
        self._net.conv1.weight.data = torch.from_numpy(np.reshape(self._genes[0:32], (32, 1, 1, 1)))
        self._net.conv1.bias.data = torch.from_numpy(self._genes[32:64])

        self._net.conv2.weight.data = torch.from_numpy(np.reshape(self._genes[64:2112], (64, 32, 1, 1)))
        self._net.conv2.bias.data = torch.from_numpy(self._genes[2112: 2176])

        self._net.conv3.weight.data = torch.from_numpy(np.reshape(self._genes[2176: 6272], (64, 64, 1, 1)))
        self._net.conv3.bias.data = torch.from_numpy(self._genes[6272:6336])

        self._net.fc1.weight.data = torch.from_numpy(np.reshape(self._genes[6336:25696448], (512, 50176)))
        self._net.fc1.bias.data = torch.from_numpy(self._genes[25696448:25696960])

        self._net.pi_logits.weight.data = torch.from_numpy(np.reshape(self._genes[25696960: 25698496], (3, 512)))
        self._net.pi_logits.bias.data = torch.from_numpy(self._genes[25698496: 25698499])

        """               
        self._net.conv1.weight.data = self._genes[0:1, :]
        self._net.conv2.weight.data = self._genes[1:33, :]
        self._net.conv3.weight.data = self._genes[33:97, :]
        self._net.fc1.weight.data = self._genes[97:50273, :]
        self._net.pi_logits.weight.data = self._genes[50273:50785, :]
        """
        #calculate fitness
        totalReward = 0
        for i in range(numberEpisodes):
            obs = self._env.reset()
            done = False
            while not done:
                action = net.forward(CNN.obs_to_torch(obs))
                obs, reward, done, _ = self._env.step(action)
                totalReward += reward
        self._fitness = totalReward / numberEpisodes
        return self._fitness

class GA:
    def __init__(self, popSize, mu, lamda, net):
        self._pop = []
        for i in range(popSize):
            self._pop.append(Chromosome(net))
        self.mu = mu
        self.lamda = lamda

    def advance(self):
        for c in self._pop:
            c.fitness(20)

        #sort(self._pop, lamda c: c._fitness, reverse = True) 
        sort(self._pop, key = c._fitness, reverse = True)
        for i in range(self.mu):
            c = self._pop[i].copy()
            c.mutate()
            self._pop[self.lamda + i] = c

    def run(self, generations):
        for i in range(generations):
            self.advance()


if __name__ == "__main__":


    net = CNN.Net()
    
    ga = GA(100, 50, 50, net) #mu = 50, lamda = 50
    ga.run(1000)