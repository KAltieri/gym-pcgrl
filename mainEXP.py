import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gym
import random
from gym_pcgrl import wrappers

import geneticAlgorithmScript
import CNN
import pandas as pd

class Chromosome:
    def __init__(self):
        self.kwargs = {
            'change_percentage': 0.4,
            'verbose': True
        }
        # 28 possibly
        self.kwargs['cropped_size'] = 28
        self.crop_size = self.kwargs.get('cropped_size', 28)
        self.agents = 1
        self.processes = 1

        self._env = wrappers.CroppedImagePCGRLWrapper("binary-narrow-v0", self.crop_size, **self.kwargs)

        n_actions = self._env.action_space.n

        #self._obs = np.zeros((1, 28, 28, 1), dtype = np.uint8)
        #self._net = CNN.Net(self._obs.shape[-1], self.crop_size, n_actions)
        self._net = CNN.Net(1, self.crop_size)
        self._fitness = 0

        self._genes = np.zeros(list(torch.flatten(self._net.conv1.weight).size())[0] + list(torch.flatten(self._net.conv1.bias).size())[0]
            + list(torch.flatten(self._net.conv2.weight).size())[0] + list(torch.flatten(self._net.conv2.bias).size())[0]
            + list(torch.flatten(self._net.conv3.weight).size())[0] + list(torch.flatten(self._net.conv3.bias).size())[0]
            + list(torch.flatten(self._net.fc1.weight).size())[0] + list(torch.flatten(self._net.fc1.bias).size())[0]
            + list(torch.flatten(self._net.pi_logits.weight).size())[0] + list(torch.flatten(self._net.pi_logits.bias).size())[0])



    def copy(self):
        c = Chromosome()
        c._genes = self._genes.copy()
        return c
    def mutation(self):
        idxlist = random.sample(range(self._genes.shape[0]), self._genes.shape[0] // 10)
        for idx in idxlist:
            random_value = np.random.normal(0, 0.1, 1) #mean of 0 and std deviation of 0.1
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
            #print(obs.shape)
            done = False
            while not done:
                action = self._net.forward(CNN.obs_to_torch(obs))
                obs, reward, done, _ = self._env.step(action)
                totalReward += reward
        self._fitness = totalReward / numberEpisodes
        #print(self._fitness)
        return self._fitness

def sortfitness(popidx):
    return popidx._fitness

def averageLstParse(lst):
    nlst = lst[:1]
    count = 0
    for i in range(len(nlst)):
        count += nlst[i]._fitness

    return count / len(nlst)

def save_models(model, path, epoch = 0, update = 0):
    print('Saving 1 model to {}'.format(path))
    save_dict = {'model_state_dict': model.state_dict()}
    if epoch:
        save_dict['epoch'] = epoch
    if update:
        save_dict['update'] = update
    print(path + 'model')
    torch.save(save_dict, path + 'model')


def load_model(path, in_channels, map_size, out_length):
    model = CNN.Net(in_channels, map_size).double()
    checkpoint = torch.load(path + 'model')
    epoch = checkpoint.get('epoch',0)
    update = checkpoint.get('update',0)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Loaded 1 models from {}.'.format(path))

    return model, epoch, update

class GA:
    def __init__(self, popSize, mu, lamda):
        self._pop = []
        for i in range(popSize):
            self._pop.append(Chromosome())
        print("done init")
        self.mu = mu
        self.lamda = lamda
        self.avgFitness = []
        #self.save_path = 'models/{}/{}'.format(game,representation)
        #self.save_path = 'runs/{}/{}/'.format("binary","narrow")
        self.save_path = 'runs/'
        self.last = False
        print(self.save_path)

    def __delete__(self, instance):
        print("deleted in descriptor object")
        del self._pop
        del self.avgFitness
        del self.mu
        del self.lamda
        return

    def advance(self):
        for c in self._pop:
            c.fitness(10)
            #c.fitness(2)
        print("done fitness")

        #sort(self._pop, lamda c: c._fitness, reverse = True)
        sorted(self._pop, key = sortfitness, reverse = True)
        self.avgFitness.append(averageLstParse(self._pop))
        save_models(self._pop[0]._net, self.save_path, epoch = 0, update = 0)
        for i in range(self.mu):
            c = self._pop[i].copy()
            c.mutation()
            self._pop[self.lamda + i] = c
        print("done mutation")

    def run(self, generations):
        for i in range(generations):
            self.last = (i == generations-1)
            self.advance()
            print("done generation")
        return self.avgFitness



if __name__ == "__main__":

    # 1 generation = 45 minutes
    allAvgFitness = []
    #ga = GA(100, 50, 50) #mu = 50, lamda = 50
    ga1= GA(2, 1, 1)
    allAvgFitness.append(ga1.run(1))
    del ga1

    print("DONE!!!")

    df = pd.DataFrame(allAvgFitness)
    df.to_csv('output.csv', index=False)
