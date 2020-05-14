import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import math

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 1, stride = 1, padding = 0)
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


## the output of our evolutionary strategies needs to be equal to the input to the convolutional neural network (maybe)
## We need a pool of chromosomes, which wouldn't work in the nested class.

def neuroEvolutionSingle(chrom, rewards):
    #Get all the weights from all the layers and flatten them
    #This will be for every chromosome (also called Neural Network or agent)

    # Conv Layers 1-3 with weights represented as floats
    weightDataConv1 = chrom.conv1.weight.data
    weightDataConv1 = weightDataConv1.numpy()

    weightDataConv2 = chrom.conv2.weight.data
    weightDataConv2 = weightDataConv2.numpy()

    weightDataConv3 = chrom.conv3.weight.data
    weightDataConv3 = weightDataConv3.numpy()

    # Fully connected layers 1-2 with weights represented as floats
    weightDataFCC1 = chrom.fc1.weight.data
    weightDataFCC1 = weightDataFCC1.numpy()

    weightDataFCC2 = chrom.value.weight.data
    weightDataFCC2 = weightDataFCC2.numpy()

    # Pass all these weights in ES script for NeuroEvolution
    weightDataConv1 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataConv1, weightDataConv1.shape[1], weightDataConv1.shape[1] * 1.5, numpy.amin(weightDataConv1), 
        numpy.amax(weightDataConv1), 40, weightDataConv1.shape[1] / 2, reward)

    weightDataConv2 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataConv2, weightDataConv2.shape[1], weightDataConv2.shape[1] * 1.5, numpy.amin(weightDataConv2), 
        numpy.amax(weightDataConv2), 40, weightDataConv2.shape[1] / 2, reward)

    weightDataConv3 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataConv3, weightDataConv3.shape[1], weightDataConv3.shape[1] * 1.5, numpy.amin(weightDataConv3), 
        numpy.amax(weightDataConv3), 40, weightDataConv3.shape[1] / 2, reward)

    weightDataFCC1 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataFCC1, weightDataFCC1.shape[1], weightDataFCC1.shape[1] * 1.5, numpy.amin(weightDataFCC1), 
        numpy.amax(weightDataFCC1), 40, weightDataFCC1.shape[1] / 2, reward)

    weightDataFCC2 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataFCC2, weightDataFCC2.shape[1], weightDataFCC2.shape[1] * 1.5, numpy.amin(weightDataFCC2), 
        numpy.amax(weightDataFCC2), 40, weightDataFCC2.shape[1] / 2, reward)

    # All values have been modified, now we need to pass them right back in!
    finalWeightDataConv1 = torch._C.from_numpy(weightDataConv1)
    chrom.conv1.weight = finalWeightDataConv1

    finalWeightDataConv2 = torch._C.from_numpy(weightDataConv2)
    chrom.conv2.weight = finalWeightDataConv2

    finalWeightDataConv3 = torch._C.from_numpy(weightDataConv3)
    chrom.conv3.weight = finalWeightDataConv3

    finalWeightDataFCC1 = torch._C.from_numpy(weightDataFCC1)
    chrom.fc1.weight = finalWeightDataFCC1

    finalWeightDataFCC2 = torch._C.from_numpy(weightDataFCC2)
    chrom.value.weight = finalWeightDataFCC2

    return chrom



def neuroEvolution(listOfChromosomes, totAccumulatedRewardList):
    #Get all the weights from all the layers and flatten them
    #This will be for every chromosome (also called Neural Network or agent)
    for i in range(len(listOfChromosomes)):
        # Both lists should be of the same size and the indices should match up
        chrom = listOfChromosomes[i]
        reward = totAccumulatedRewardList[i]

        # Conv Layers 1-3 with weights represented as floats

        # Make sure to include the biases with the data (look this up whether or not it's included)

        weightDataConv1 = chrom.conv1.weight.data
        weightDataConv1 = weightDataConv1.numpy()

        weightDataConv2 = chrom.conv2.weight.data
        weightDataConv2 = weightDataConv2.numpy()

        weightDataConv3 = chrom.conv3.weight.data
        weightDataConv3 = weightDataConv3.numpy()

        # Fully connected layers 1-2 with weights represented as floats
        weightDataFCC1 = chrom.fc1.weight.data
        weightDataFCC1 = weightDataFCC1.numpy()

        weightDataFCC2 = chrom.value.weight.data
        weightDataFCC2 = weightDataFCC2.numpy()

        # Pass all these weights in ES script for NeuroEvolution
        weightDataConv1 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataConv1, weightDataConv1.shape[1], weightDataConv1.shape[1] * 1.5, numpy.amin(weightDataConv1), 
            numpy.amax(weightDataConv1), 40, weightDataConv1.shape[1] / 2, reward)

        weightDataConv2 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataConv2, weightDataConv2.shape[1], weightDataConv2.shape[1] * 1.5, numpy.amin(weightDataConv2), 
            numpy.amax(weightDataConv2), 40, weightDataConv2.shape[1] / 2, reward)

        weightDataConv3 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataConv3, weightDataConv3.shape[1], weightDataConv3.shape[1] * 1.5, numpy.amin(weightDataConv3), 
            numpy.amax(weightDataConv3), 40, weightDataConv3.shape[1] / 2, reward)

        weightDataFCC1 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataFCC1, weightDataFCC1.shape[1], weightDataFCC1.shape[1] * 1.5, numpy.amin(weightDataFCC1), 
            numpy.amax(weightDataFCC1), 40, weightDataFCC1.shape[1] / 2, reward)

        weightDataFCC2 = geneticAlgorithmScript.geneticAlgorithmMain(weightDataFCC2, weightDataFCC2.shape[1], weightDataFCC2.shape[1] * 1.5, numpy.amin(weightDataFCC2), 
            numpy.amax(weightDataFCC2), 40, weightDataFCC2.shape[1] / 2, reward)

        # All values have been modified, now we need to pass them right back in!
        finalWeightDataConv1 = torch._C.from_numpy(weightDataConv1)
        chrom.conv1.weight = finalWeightDataConv1

        finalWeightDataConv2 = torch._C.from_numpy(weightDataConv2)
        chrom.conv2.weight = finalWeightDataConv2

        finalWeightDataConv3 = torch._C.from_numpy(weightDataConv3)
        chrom.conv3.weight = finalWeightDataConv3

        finalWeightDataFCC1 = torch._C.from_numpy(weightDataFCC1)
        chrom.fc1.weight = finalWeightDataFCC1

        finalWeightDataFCC2 = torch._C.from_numpy(weightDataFCC2)
        chrom.value.weight = finalWeightDataFCC2

    return listOfChromosomes



        

        
        


        
    

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