from math import log
import numpy as np
from nltk import RegexpParser
from nltk.tree import Tree
import re
import time
import matplotlib.pyplot as plt

from nltk import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from hatesonar import Sonar
from pympler import asizeof

class NaiveBayesClassifier: #parent naive bayes classifier
    def buildModel(self, infile):
        pass

    def fitModelNaiveBayes(self, k=1):
        pass

    def predictAndFindAccuracy(self, infile):
        pass

    def laplaceSmoothing(self, infile, maxK):
        kvalues = []
        for i in range(0, maxK+1):
            kvalues.append(float(i)/float(maxK))
        results = []
        for k in kvalues:
            self.fitModelNaiveBayes(k)
            pr = self.predictAndFindAccuracy(infile)
            results.append(pr)

        plt.plot(kvalues, results)
        plt.title("Accuracy vs K for Naive Bayes Classifier")
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.show()

        index = np.argmax(results)
        print ("ACCURACY WITH LAPLACE SMOOTHING: ", np.max(results))
        return kvalues[index]

    def runClassifier(self, maxK):
        start = time.time()
        print ("Processing training set...")
        self.buildModel('mini.train')
        if hasattr(self, 'dict'):
            print (len(self.dict), "words in dictionary")
        print ("Fitting model...")
        self.fitModelNaiveBayes()
        print ("Accuracy on validation set:", self.predictAndFindAccuracy('mini.valid'))
        end = time.time()
        print ("TIME:", (end - start))
        print ("MEMORY:", (asizeof.asizeof(self)))
        print ("Good k for Laplace:", self.laplaceSmoothing('mini.valid', maxK))

class BagOfWordsClassifier(NaiveBayesClassifier):
    def buildModel(self, infile):
        """
        Assume infile has one review per line,
            starting with the rating: whether sentence is gendered (1) or not (0),
            and then followed with a space.
        Ignore punctuation and capitalization of each word.
        Build the dictionary of word occurences,
            count the occurrences of each word in each rating,
            and count the number of reviews with each rating.
        counts[rating][word] is the number of times word appears in any of the
        sentences corresponding to the rating
        nrated[rating] is the total number of sentences either gendered or not
        """

        self.dict = {}
        self.nrated = [0] * 2 #binary either gendered or not

        with open(infile,'r') as f:
            curr_index = 0
            for line in f.readlines():
                for index, word in enumerate(line.lower().split()):
                    if index == 0:
                        rating = int(word)
                        self.nrated[rating] += 1
                    else:
                        word = word.strip('.')
                        if self.dict.get(word) == None:
                            self.dict[word] = curr_index
                            curr_index += 1

        # Fill counts
        self.counts = [[0] * len(self.dict) for _ in range(2)]

        with open(infile, 'r') as f:
            for line in f.readlines():
                line_content = line.lower().split()
                rating = int(line_content[0])
                for word in line_content[1:]:
                    word = word.strip('.')
                    index = self.dict.get(word)
                    self.counts[rating][index] += 1

    def fitModelNaiveBayes(self, k=1):
        """
        F[rating][word] = -log(p(word|rating)).

        P(word|rating) =
        (k + self.counts[rating][word]) /
          (sum(self.counts[rating]) + (k * len(self.counts[rating])))
        """
        self.F = [[0] * len(self.dict) for _ in range(2)]

        for rating in range(2):
            for word in self.dict:
                word = word.strip('.')
                word_index = self.dict[word]
                p = float(k + self.counts[rating][word_index]) \
                    / (float(sum(self.counts[rating])) + \
                     float((k*len(self.counts[rating])))) #P(Wi|Y)
                if p == 0:
                    self.F[rating][word_index] = 0
                else:
                    self.F[rating][word_index] = -log(p)
            
    def predictAndFindAccuracy(self, infile):
        """
        Given a test data file, returns accuracy of predictions.
        P(Y| W1...Wn) = product of P(Wi|Y), so we add up all the values in self.F
            (which has log of the probability)
        """
        predicted_ratings = []
        actual_ratings = []

        with open(infile, 'r') as f:
            for line in f.readlines():
                rating = int(line.split()[0])
                word_ratings = []
                for i in range(2):
                    sent = 0
                    for word in line.lower().split()[1:]:
                        word = word.strip('.')
                        if self.dict.get(word) != None:
                            sent += self.F[i][self.dict[word]]
                    word_ratings.append(sent)
                actual_ratings.append(rating)
                predicted_ratings.append(range(2)[np.argmin(word_ratings)])
            
        correct = 0
        for i in range(len(predicted_ratings)):
            if predicted_ratings[i] == actual_ratings[i]:
                correct += 1
        return float(correct)/float(len(predicted_ratings))

class BigramClassifier(NaiveBayesClassifier):
    '''
    similar to bag of words, but instead record pairs of ordered words in dictionary
    '''
    def buildModel(self, infile):
        self.dict = {}
        self.nrated = [0] * 2 #binary either gendered or not

        with open(infile,'r') as f:
            curr_index = 0
            for line in f.readlines():
                line_content = line.lower().split()
                rating = int(line_content[0])
                self.nrated[rating] += 1
                for i in range(1, len(line_content)-1):
                    bigram = (line_content[i].strip('.'), \
                        line_content[i + 1].strip('.'))
                    if self.dict.get(bigram) == None:
                        self.dict[bigram] = curr_index
                        curr_index += 1
        # Fill counts
        self.counts = [[0] * len(self.dict) for _ in range(2)]

        with open(infile, 'r') as f:
            for line in f.readlines():
                line_content = line.lower().split()
                rating = int(line_content[0])
                for i in range(1, len(line_content)-1):
                    bigram = (line_content[i].strip('.'), line_content[i + 1].strip('.'))
                    index = self.dict.get(bigram)
                    self.counts[rating][index] += 1

    def fitModelNaiveBayes(self, k=1):
        self.F = [[0] * len(self.dict) for _ in range(2)]

        for rating in range(2):
            for bigram in self.dict:
                bigram_index = self.dict[bigram]
                p = float(k + self.counts[rating][bigram_index]) \
                    / (float(sum(self.counts[rating])) + \
                     float((k*len(self.counts[rating])))) #P(Wi|Y)
                if p == 0:
                    self.F[rating][bigram_index] = 0
                else:
                    self.F[rating][bigram_index] = -log(p)
            
    def predictAndFindAccuracy(self, infile):
        predicted_ratings = []
        actual_ratings = []

        with open(infile, 'r') as f:
            for line in f.readlines():
                line_content = line.lower().split()
                rating = int(line_content[0])
                bigram_ratings = []
                for rating in range(2):
                    sent = 0
                    for i in range(1, len(line_content)-1):
                        bigram = (line_content[i].strip('.'), line_content[i + 1].strip('.'))
                        if self.dict.get(bigram) != None:
                            sent += self.F[rating][self.dict[bigram]]
                    bigram_ratings.append(sent)
                actual_ratings.append(rating)
                predicted_ratings.append(range(2)[np.argmin(bigram_ratings)])
            
        correct = 0
        for i in range(len(predicted_ratings)):
            if predicted_ratings[i] == actual_ratings[i]:
                correct += 1
        return float(correct)/float(len(predicted_ratings))

class FeaturesClassifier(NaiveBayesClassifier):
    def loadBannedWords(self): # used online article for list of gendered words
        self.bannedWords = []
        with open("genderedwords.txt", 'r') as f:
            for line in f.readlines():
                self.bannedWords.append(line.lower())
    def countBannedWords(self, line): 
        self.loadBannedWords()
        count = 0
        for index, word in enumerate(line.split()):
            if word.lower().strip('.') in self.bannedWords:
                count += 1
        return count

    def countExclamationPoints(self, line):
        count = 0
        for word in line.split():
            if '!' in word:
                count += 1
        return count

    def countCommands(self, line): # check if first line of sentence is verb
        count = 0
        for sentence in filter(None, re.split(r"[.!?\-]+", line)):
            words = sentence.split()
            if words:
                firstWord = word_tokenize(words[0])
                if nltk.pos_tag(firstWord)[0][1] == 'VB':
                    count += 1
        return count

    def isHateSpeech(self, line): #using open source hate sonar api
        sonar = Sonar()
        response = sonar.ping(text=line)
        if response["top_class"] != "neither": # line is hate speech
            return 1
        else:
            return 0

    def buildModel(self, infile):
        self.nrated = [0] * 2 #binary either gendered or not
        with open(infile,'r') as f:
            for line in f.readlines():
                for index, word in enumerate(line.lower().split()):
                    if index == 0:
                        rating = int(word)
                        self.nrated[rating] += 1

        self.featureFreq = [{} for _ in range(2)]
        with open(infile, 'r') as f:
            for line in f.readlines():
                rating = int(line[0])
                sentence = line[2:].lower()
                numBannedWords = self.countBannedWords(sentence)
                numExclPoints = self.countExclamationPoints(sentence)
                numCommands = self.countCommands(sentence)
                isHateSpeech = self.isHateSpeech(sentence)

                bannedStr = 'banned' + str(numBannedWords)
                exclPointsStr = 'exclPt' + str(numExclPoints)
                commandsStr = 'command' + str(numCommands)
                isHateStr = 'isHate' + str(isHateSpeech)

                featureDict = self.featureFreq[rating]

                if bannedStr not in featureDict:
                    featureDict[bannedStr] = 0
                if exclPointsStr not in featureDict:
                    featureDict[exclPointsStr] = 0
                if commandsStr not in featureDict:
                    featureDict[commandsStr] = 0
                if isHateStr not in featureDict:
                    featureDict[isHateStr] = 0

                featureDict[bannedStr] += 1
                featureDict[exclPointsStr] += 1
                featureDict[commandsStr] += 1
                featureDict[isHateStr] += 1

    def totalSentencesForFeature(self, featureName, rating):
        count = 0
        for key in self.featureFreq[rating]:
            if featureName in key:
                count += self.featureFreq[rating][key]
        return count

    def totalOutcomesForFeature(self, featureName, rating):
        count = 0
        for key in self.featureFreq[rating]:
            if featureName in key:
                count += 1
        return count

    def fitModelNaiveBayes(self, k=0):
        self.F = [{} for _ in range(2)]

        for rating in range(2):
            for feature_i in self.featureFreq[rating]:
                featureName = ''.join([i for i in feature_i if not i.isdigit()])
                p = float(k + self.featureFreq[rating][feature_i]) \
                    / (self.totalSentencesForFeature(featureName, rating) + \
                     k*self.totalOutcomesForFeature(featureName, rating)) #P(Fi|Y)
                if p == 0:
                    self.F[rating][feature_i] = 0
                else:
                    self.F[rating][feature_i] = -log(p)
            
    def predictAndFindAccuracy(self, infile):
        predicted_ratings = []
        actual_ratings = []

        with open(infile, 'r') as f:
            for line in f.readlines():
                rating = int(line.split()[0])
                word_ratings = []
                for i in range(2):
                    sent = 0

                    sentence = line[2:].lower()
                    numBannedWords = self.countBannedWords(sentence)
                    numExclPoints = self.countExclamationPoints(sentence)
                    numCommands = self.countCommands(sentence)
                    isHateSpeech = self.isHateSpeech(sentence)

                    bannedStr = 'banned' + str(numBannedWords)
                    exclPointsStr = 'exclPt' + str(numExclPoints)
                    commandsStr = 'command' + str(numCommands)
                    isHateStr = 'isHate' + str(isHateSpeech)

                    # P(Y| F1...Fn) = product of P(Fi|Y), so add logs
                    if bannedStr in self.F[i]:
                        sent += self.F[i][bannedStr]
                    if exclPointsStr in self.F[i]:
                        sent += self.F[i][exclPointsStr]
                    if commandsStr in self.F[i]:
                        sent += self.F[i][commandsStr]
                    if isHateStr in self.F[i]:
                        sent += self.F[i][isHateStr]

                    word_ratings.append(sent)
                actual_ratings.append(rating)
                predicted_ratings.append(range(2)[np.argmin(word_ratings)])
            
        correct = 0
        for i in range(len(predicted_ratings)):
            if predicted_ratings[i] == actual_ratings[i]:
                correct += 1
        return float(correct)/float(len(predicted_ratings))

if __name__ == '__main__':
    print ("BAG OF WORDS NAIVE BAYES CLASSIFIER")
    c = BagOfWordsClassifier()
    c.runClassifier(100)

    print ("BIGRAM NAIVE BAYES CLASSIFIER")
    c = BigramClassifier()
    c.runClassifier(100)

    print ("FEATURE NAIVE BAYES CLASSIFIER") # warning: takes a while to run
    c = FeaturesClassifier()
    c.runClassifier(10)