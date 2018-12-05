from math import log
import numpy as np
from nltk import RegexpParser
from nltk.tree import Tree
import re
import time

from nltk import word_tokenize
import nltk
from hatesonar import Sonar
from pympler import asizeof

#TO DO:
# - integrate with sample data (train and test) RACHEL DONE

# - naive bayes for other 4 features RACHEL DONE
#   - TO BE DONE AS CLEANUP: don't hard code features into NB RACHEL (after Friday)
# - different word models RACHEL (after Friday)

# - laplace smoothing KOFI
# - k nearest neighbors KOFI
#   - word vector vs library vs features

#DIFFERENT ALGORITHMS: each bullet point (return accuracy, runtime, memory)
# - Naive Bayes with Bag of Words
#   - Laplace Smoothing KOFI
# - Naive Bayes with features
#   - Laplace Smoothing KOFI
# - Naive Bayes with bigram/tfidf
#   - Laplace Smoothing KOFI
# - K nearest neighbors KOFI
#   - word vector
#   - library
#   - features

# - to consider: feature hashing?

class OtherFeaturesClassifier:
    def loadBannedWords(self):
        self.bannedWords = []
        with open("genderedwords.txt", 'r') as f:
            for line in f.readlines():
                self.bannedWords.append(line.lower())

    def countBannedWords(self, line): 
        self.loadBannedWords()
        count = 0
        for index, word in enumerate(line.split()):
            if word.lower() in self.bannedWords:
                count += 1
        return count

    def countExclamationPoints(self, line):
        count = 0
        for word in line.split():
            if word == '!':
                count += 1
        return count

    def countCommands(self, line):
        count = 0
        for sentence in filter(None, re.split(r"[.!?\-]+", line)):
            words = sentence.split()
            if words:
                firstWord = word_tokenize(words[0])
                if nltk.pos_tag(firstWord)[0][1] == 'VB':
                    count += 1
        return count

    #def countNameCalling(self, line):

    def isHateSpeech(self, line): #using open source hate sonar api
        #indices = {"hate_speech": 0, "offensive_language": 1, "neither": 2}
        sonar = Sonar()
        response = sonar.ping(text=line)
        #indexOfLanguage = indices[response["top_class"]]
        if response["top_class"] != "neither":
            #return response['classes'][indexOfLanguage]['confidence']
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

        # Fill counts
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

    def fitModelNaiveBayes(self, alpha=1):
        """
        Now you'll fit the model. For historical reasons, we'll call it F.
        F[rating][word] is -log(p(word|rating)).

        P(word|rating) =
        (alpha + self.counts[rating][word]) /
          (sum(self.counts[rating]) + (alpha * len(self.counts[rating])))
        """
        self.F = [{} for _ in range(2)]

        for rating in range(2):
            for feature_i in self.featureFreq[rating]:
                featureName = ''.join([i for i in feature_i if not i.isdigit()])
                p = float(alpha + self.featureFreq[rating][feature_i]) \
                    / (self.totalSentencesForFeature(featureName, rating) + \
                     self.totalOutcomesForFeature(featureName, rating)) #P(Fi|Y)
                if p == 0:
                    self.F[rating][feature_i] = 0
                else:
                    self.F[rating][feature_i] = -log(p)
            
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

                    if bannedStr in self.F[i]:
                        sent += self.F[i][bannedStr] # P(Y| F1...Fn) = product of P(Fi|Y)
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


class BagOfWordsClassifier:
    def buildModel(self, infile):
        """
        The infile has one review per line, starting with the rating and then a space.
        Note that the "words" include things like punctuation and numbers. Don't worry
        about this distinction for now; any string that occurs between spaces is a word.

        You must do three things in this question: build the dictionary,
        count the occurrences of each word in each rating and count the number
        of reviews with each rating.
        The words should be numbered sequentially in the order they first appear.
        counts[ranking][word] is the number of times word appears in any of the
        reviews corresponding to ranking
        nrated[ranking] is the total number of reviews with each ranking

        Hint: Make sure to actually set the self.dict, self.counts, and
        self.nrated variables!
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
                     float((alpha*len(self.counts[rating])))) #P(Wi|Y)
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
                for i in range(2):
                    sent = 0
                    for word in line.lower().split()[1:]:
                        if self.dict.get(word) != None:
                            sent += self.F[i][self.dict[word]] # P(Y| W1...Wn) = product of P(Wi|Y)
                    word_ratings.append(sent)
                actual_ratings.append(rating)
                predicted_ratings.append(range(2)[np.argmin(word_ratings)])
            
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
    # print ("BAG OF WORDS NAIVE BAYES CLASSIFIER")
    # bagOfWordsStart = time.time()
    # c = BagOfWordsClassifier()
    # print ("Processing training set...")
    # c.buildModel('mini.train')
    # print (len(c.dict), "words in dictionary")
    # print ("Fitting model...")
    # c.fitModelNaiveBayes()
    # print ("Accuracy on validation set:", c.predictAndFindAccuracy('mini.valid')[1])
    # bagOfWordsEnd = time.time()
    # print ("Good alpha:", c.tuneAlpha('mini.valid'))
    # print ("TIME:", (bagOfWordsEnd - bagOfWordsStart))
    # print ("MEMORY:", (asizeof.asizeof(c)))

    print ("FEATURE NAIVE BAYES CLASSIFIER")
    featuresStart = time.time()
    c = OtherFeaturesClassifier()
    print ("Processing training set...")
    c.buildModel('mini.train')
    print ("Fitting model...")
    c.fitModelNaiveBayes()
    print ("Accuracy on validation set:", c.predictAndFindAccuracy('mini.valid')[1])
    featuresEnd = time.time()
    print ("TIME:", (featuresEnd - featuresStart))
    print ("MEMORY:", (asizeof.asizeof(c)))