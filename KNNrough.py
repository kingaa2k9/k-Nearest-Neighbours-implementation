import csv
import random
import operator
import math


def loadDataset(filename, split, trainingset=[], testset=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingset.append(dataset[x])
            else:
                testset.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingset, testinstance, k):
    distances = []
    length = len(testinstance) - 1
    for x in range(len(trainingset)):
        dist = euclideanDistance(testinstance, trainingset[x], length)
        distances.append((trainingset[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testset, predictions):
    correct = 0
    for x in range(len(testset)):
        if testset[x][-1] == predictions[x]:
            correct += 1
    return float((correct / len(testset))) * 100.0


def main():
    trainingset = []
    testset = []
    split = 0.67
    loadDataset(r'C:\Users\DIPIN_PC\Desktop\ML\iris.data.txt', split, trainingset, testset)
    print('Train:' + repr(len(trainingset)))
    print('Test:' + repr(len(testset)))
    predictions = []
    k = 3
    for x in range(len(testset)):
        neighbors = getNeighbors(trainingset, testset[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predicted=' + repr(result) + ',         actual=' + repr(testset[x][-1]))
    accuracy = getAccuracy(testset, predictions)
    print('Accuracy:' + repr(accuracy) + '%')


main()