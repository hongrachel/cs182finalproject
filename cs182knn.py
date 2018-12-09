from math import log
import numpy as np
from nltk import RegexpParser
from nltk.tree import Tree
import re
import time
import smart_open

from nltk import word_tokenize
import nltk
from hatesonar import Sonar
from pympler import asizeof

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class knnClassifier:

    def __init__(self, k):
        self.K = k 
        self.trainingLabels = [[],[]]
        self.testingLabels = [[],[]]

    def predictAndComputeAccuracy(self, infile):
        '''Determine the accuracy of assignment of test file based on trained data'''
        predictedClassification = []
        actualClassification = []

        points = []
        for point in self.trainingLabels[0]:
            points.append((point, 0))
        for point in self.trainingLabels[1]:
            points.append((point, 1))

        with open(infile, 'r') as f:
            for line in f.readlines():
                actualRating = line[0]
                sentence = line[2:].strip("\n")

                actualClassification.append((sentence, int(actualRating)))
                
                sentenceVector = self.vectorize(sentence)

                # Take the first k distances
                kDistances = sorted([(np.linalg.norm(np.subtract(point, (sentenceVector))), rating) \
                    for (point, rating) in points], key=lambda x: x[0])[:self.K]
                # Determine the probability of being gendered by averaging
                genderedLikelihood = sum([n for _, n in kDistances])/float(self.K)

                # TODO Break ties by distances
                if genderedLikelihood <= 0.5:
                    predictedClassification.append((sentence, 0))
                    ## For graphing purposes; only relevant in features classification
                    self.testingLabels[0].append(sentenceVector)
                else:
                    predictedClassification.append((sentence, 1))
                    self.testingLabels[1].append(sentenceVector)
            
        correct = 0
        for i in range(len(predictedClassification)):
            # Check if each sentence was given the same rating
            if predictedClassification[i][1] == actualClassification[i][1]:
                correct += 1

        accuracy = correct/float(len(predictedClassification))
        return (accuracy)

    
    def graphScatter(self, labels, dataType):
        #scale = 50.0
        #scale = 100.0 * np.random.rand(len(labels[0]+labels[1]))

        for index, label in enumerate(labels):
            x, y = zip(*label)
            color, label = "black", "n/a"

            if index == 0: # Non-gendered
                color, label = "green", "Non-gendered"
                scale = 100.0
            else: # gendered
                color, label = "red", "Gendered"
                scale = 50
            try:
                plt.scatter(x, y, alpha=0.1, s=scale, c=color, label=label)
            except:
                continue
        plt.title('K-NN: {} Data, K={}'.format(dataType, K))
        plt.xlabel = "Hatespeech Score"
        plt.ylabel = "Number of banned words"
        plt.legend()
        plt.show()   

class wordVectorClassifier(knnClassifier):
    '''Classify sentences according to vectors based on existing words'''
    def __init__(self, k):
        super().__init__(k)

    def loadTrainingData(self, infile):
        ''' 
        Load Training Data using vectors similar to the bag of words model
        '''
        self.dict = {}

        # Load the dictionary
        with open(infile,'r') as f:
            curr_index = 0
            for line in f.readlines():
                for index, word in enumerate(line.lower().split()):
                    if index == 0:
                        continue
                    else:
                        if self.dict.get(word) == None:
                            self.dict[word] = curr_index
                            curr_index += 1

        with open(infile, 'r') as f:
            for line in f.readlines():
                rating = int(line[0])
                sentence = line[2:].lower().strip("\n")

                self.trainingLabels[rating].append(self.vectorize(sentence))

    def vectorize(self,sentence):
        ''' Vectorize a given sentence according the bag of words model'''
        vector = [0]*len(self.dict)
        for word in str(sentence).split():
            word = self.dict.get(word.lower())
            if word:
                vector[word] = 1

        return vector 

class libraryVectorClassifier(knnClassifier):
    '''Classify sentences through the use of a library that incorporates sentence meaning into vector'''
    def __init__(self, k):
        super().__init__(k)

    def loadTrainingData(self, infile):
        '''Create model to vectorize sentences based on training data using gensim library'''
        sentences = []
        # Create the model to vectorize sentences
        with open(infile, 'r') as f:
            for line in f.readlines():
                sentence = line[2:].lower().strip(". \n")
                sentences.append(sentence.split(" "))

        document = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
        self.model = Doc2Vec(document, vector_size=100, window=2, min_count=1, workers=4)

        # Label sentences as vectors
        with open(infile, 'r') as f:
            for line in f.readlines():
                rating = int(line[0])
                sentence = line[2:].lower().strip(". \n").split(" ")

                self.trainingLabels[rating].append(self.vectorize(sentence))

    def vectorize(self, sentence): 
        '''Vectorize a sentence based on the model'''
        return self.model.infer_vector(sentence)

class featureClassifier(knnClassifier):
    '''Classify sentences based on features - x: hate speech score, y: number of banned words'''
    def __init__(self, k):
        super().__init__(k)
        self.loadBannedWords()

    def loadBannedWords(self):
        '''load banned words into class'''
        self.bannedWords = set()
        with open("genderedwords.txt", 'r') as f:
            for line in f.readlines():
                self.bannedWords.add(line.lower().strip("\n"))  
    
    def countBannedWords(self, line):
        '''Count the number of banned words within a sentence''' 
        count = 0
        for index, word in enumerate(line.split()):
            if word.lower().strip(".") in self.bannedWords:
                count += 1
        return count  

    def isHateSpeech(self, line):
        '''Assign a 'hatespeech score' using sonar api '''
        indices = {"hate_speech": 0, "offensive_language": 1, "neither": 2}
        sonar = Sonar()
        response = sonar.ping(text=line)
        indexOfLanguage = indices[response["top_class"]]
        if response["top_class"] != "neither":
            return response['classes'][indexOfLanguage]['confidence']
        else:
            return 0

    def vectorize(self, sentence):
        '''Vectorize sentence to (x,y) according to (hateSpeechScore, bannedWordCount)'''
        x = self.isHateSpeech(sentence)
        y = self.countBannedWords(sentence)

        return (x,y)

    def loadTrainingData(self, infile):
        '''Classify training data'''
        with open(infile,'r') as f:
            for line in f.readlines():

                rating = line[0]
                sentence = line[2:].lower().strip(". \n")

                vector = self.vectorize(sentence)

                self.trainingLabels[int(rating)].append(vector)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print ("K-Nearest Neighbors Classifications")

    K = 5   
    # '''Classify based on word vectors'''
    # print ("Word Vector Classifier")
    # classifier = wordVectorClassifier(K)
    # ### Training ###
    # print ("Processing training set...")
    # trainingStart = time.time()
    # classifier.loadTrainingData('mini.train')
    # trainingEnd = time.time()

    # trainingTime = trainingEnd-trainingStart
    # trainingSize = asizeof.asizeof(classifier)

    # ### Testing ###
    # print ("Processing test set...")
    # testingStart = time.time()
    # accuracy = classifier.predictAndComputeAccuracy('mini.valid')
    # testingEnd = time.time()

    # testingTime = testingEnd - testingStart
    # testingSize = asizeof.asizeof(classifier)

    # print("Accuracy: ", accuracy)
    # print("Time on Training: ", trainingTime)
    # print("Time on Testing: ", testingTime)
    # print("Training Memory: ", trainingSize)
    # print("Testing Memory: ", testingSize)

    # '''Classify based on sentence meaning'''
    # classifier = libraryVectorClassifier(K)
    # ### Training ###
    # print ("Library Vectors Classifier")
    # print ("Processing training set...")
    # trainingStart = time.time()
    # classifier.loadTrainingData('mini.train')
    # trainingEnd = time.time()

    # trainingTime = trainingEnd-trainingStart
    # trainingSize = asizeof.asizeof(classifier)

    # ### Testing ###
    # print ("Processing test set...")
    # testingStart = time.time()
    # accuracy = classifier.predictAndComputeAccuracy('mini.valid')
    # testingEnd = time.time()

    # testingTime = testingEnd - testingStart
    # testingSize = asizeof.asizeof(classifier)

    # print("Accuracy: ", accuracy)
    # print("Time on Training: ", trainingTime)
    # print("Time on Testing: ", testingTime)
    # print("Training Memory: ", trainingSize)
    # print("Testing Memory: ", testingSize)


    '''Classify based on features'''
    print ("feature vector classifier")
    classifier = featureClassifier(K)
    ### Training ###
    print ("Processing training set...")
    trainingStart = time.time()
    classifier.loadTrainingData('mini.train')
    trainingEnd = time.time()

    trainingTime = trainingEnd-trainingStart
    trainingSize = asizeof.asizeof(classifier)

    classifier.graphScatter(classifier.trainingLabels, "Training")

    ### Testing ###
    print ("Processing test set...")
    testingStart = time.time()
    accuracy = classifier.predictAndComputeAccuracy('mini.valid')
    testingEnd = time.time()

    testingTime = testingEnd - testingStart
    testingSize = asizeof.asizeof(classifier)

    classifier.graphScatter(classifier.testingLabels, "Testing")

    print("Accuracy: ", accuracy)
    print("Time on Training: ", trainingTime)
    print("Time on Testing: ", testingTime)
    print("Training Memory: ", trainingSize)
    print("Testing Memory: ", testingSize)