from math import log
import numpy as np
from nltk import RegexpParser
from nltk.tree import Tree
import re

from nltk import word_tokenize
from hatesonar import Sonar

#TO DO:
# - integrate with real data (train and test) KOFI?
# - naive bayes for other features RACHEL
# - different word models RACHEL
# - laplace smoothing KOFI
# - parameter estimation KOFI
# - k nearest neighbors RACHEL

class OtherFeatures:
    def loadBannedWords(self):
        self.bannedWords = []
        with open("genderedwords.txt", 'r') as f:
            for line in f.readlines():
                self.bannedWords.append(line.lower())

    def countBannedWords(self, line): 
        self.loadBannedWords()
        count = 0
        for index, word in enumerate(line.split()):
            if word.lower() in bannedWords:
                count += 1
        return count

    def countExplanationPoints(self, line):
        count = 0
        for word in line.split():
            if word == '!':
                count += 1
        return count

    def countCommands(self, line):
        count = 0
        for sentence in filter(None, re.split(r"[.!?\-]+", line)):
            firstWord = word_tokenize(sentence[1])
            if nltk.pos_tag(firstWord)[0][1] == 'VB':
                count += 1
        return count 

    def scoreHateSpeech(self, line): #using open source hate sonar api
        indices = {"hate_speech": 0, "offensive_language": 1, "neither": 2}
        sonar = Sonar()
        response = sonar.ping(text=line)
        indexOfLanguage = indices[response["top_class"]]
        if response["top_class"] != "neither":
            return response['classes'][indexOfLanguage]['confidence']
        else:
            return 0


class BagOfWordsClassifier:
    def buildBagOfWords(self, infile):
        self.dict = {}
        self.nrated = [0] * 2 #binary either gendered or not

        with open(infile,'r') as f:
            curr_index = 0
            for line in f.readlines():
                for index, word in enumerate(line.split()):
                    if index == 0:
                        rating = int(word)
                        self.nrated[rating] += 1
                    else:
                        if self.dict.get(word) == None:
                            self.dict[word] = curr_index
                            curr_index += 1

        # Fill counts
        self.counts = [[0] * len(self.dict) for _ in range(2)]

        with open(infile, 'r') as f:
            for line in f.readlines():
                line_content = line.split()
                rating = int(line_content[0])
                for word in line_content[1:]:
                    index = self.dict.get(word)
                    self.counts[rating][index] += 1

    def fitModelNaiveBayes(self, alpha=1):
        """
        Now you'll fit the model. For historical reasons, we'll call it F.
        F[rating][word] is -log(p(word|rating)).

        P(word|rating) =
        (alpha + self.counts[rating][word]) /
          (sum(self.counts[rating]) + (alpha * len(self.counts[rating])))
        """
        self.F = [[0] * len(self.dict) for _ in range(2)]

        for rating in range(2):
            for word in self.dict:
                word_index = self.dict[word]
                p = float(alpha + self.counts[rating][word_index]) \
                    / (float(sum(self.counts[rating])) + \
                     float((alpha*len(self.counts[rating]))))
                if p == 0:
                    self.F[rating][word_index] = 0
                else:
                    self.F[rating][word_index] = -log(p)
            
    def predictAndFindAccuracy(self, infile):
        """
        Test time! The infile has the same format as it did before. For each review,
        predict the rating. Ignore words that don't appear in your dictionary.
        Are there any factors that won't affect your prediction?
        You'll report both the list of predicted ratings in order and the accuracy.
        """
        predicted_ratings = []
        actual_ratings = []

        with open(infile, 'r') as f:
            for line in f.readlines():
                rating = int(line.split()[0])
                word_ratings = []
                for i in range(5):
                    sent = 1
                    for word in line.split()[1:]:
                        if self.dict.get(word) != None:
                            sent *= self.F[i][self.dict[word]]
                    word_ratings.append(sent)
                actual_ratings.append(rating)
                predicted_ratings.append(range(5)[np.argmin(word_ratings)])
            
        correct = 0
        for i in range(len(predicted_ratings)):
            if predicted_ratings[i] == actual_ratings[i]:
                correct += 1
        return (predicted_ratings, float(correct)/float(len(predicted_ratings)))

    def tuneAlpha(self, infile):
        """
        Alpha is a hyperparameter of this model - a tunable option that affects
        the values that appear in F. Let's tune it!
        We've split the dataset into 3 parts: the training set you use to fit the model
        the validation and test sets you use to evaluate the model. The training set
        is used to optimize the regular parameters, and the validation set is used to
        optimize the hyperparameters. (Why don't you want to set the hyperparameters
        using the test set accuracy?)
        Find and return a good value of alpha
        """
        accuracies = []
        for i in range(1, 1001):
            accuracies.append(float(i)/float(1000))

        print(accuracies)
        results = []
        for a in accuracies:
            self.fitModelNaiveBayes(a)
            _, pr = self.predictAndFindAccuracy(infile)
            results.append(-pr)

        index = np.argmin(results)
        return accuracies[index]


if __name__ == '__main__':
    c = BagOfWordsClassifier()
    print "Processing training set..."
    c.buildBagOfWords('mini.train')
    print len(c.dict), "words in dictionary"
    print "Fitting model..."
    c.fitModelNaiveBayes()
    print "Accuracy on validation set:", c.predictAndFindAccuracy('mini.valid')[1]
    print "Good alpha:", c.tuneAlpha('mini.valid')