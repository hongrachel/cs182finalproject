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

# from gensim.models import Word2Vec

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

class knnClassification:

    def __init__(self, k):
        self.K = k 

    def loadBannedWords(self):
        self.bannedWords = []
        with open("genderedwords.txt", 'r') as f:
            for line in f.readlines():
                self.bannedWords.append(line.lower().strip("\n"))   

    def countBannedWords(self, line): 
        self.loadBannedWords()
        count = 0
        for index, word in enumerate(line.split()):
            if word.lower().strip(".") in self.bannedWords:
                count += 1
        return count    

    def isHateSpeech(self, line): #using open source hate sonar api
        indices = {"hate_speech": 0, "offensive_language": 1, "neither": 2}
        sonar = Sonar()
        response = sonar.ping(text=line)
        indexOfLanguage = indices[response["top_class"]]
        if response["top_class"] != "neither":
            return response['classes'][indexOfLanguage]['confidence']
        else:
            return 0

    def loadTrainingDataUsingFeatures(self, infile):
        # Translate sentences into points (x,y) -> (hateSpeechScore, bannedWords) based on features
        # Use nrated to keep track of the number of sentence values that we have
        self.nrated = [0] * 2 # binary either gendered or not
        self.labels = [[], []] # binary gendered [0] or not [1]
        self.labelsTest = [[],[]]
        with open(infile,'r') as f:
            for line in f.readlines():

                rating = line[0]
                sentence = line[2:]

                x = self.isHateSpeech(line[2:])
                y = self.countBannedWords(line[2:])
                ## Group the sentence
                self.labels[int(rating)].append((x,y))
                
    def loadTrainingDataUsingVectors(self, infile):
        self.dict = {}
        self.labels = [[],[]]

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
                sentence = line[2:].strip("\n")

                self.labels[rating].append(self.vectorize(sentence))

    # vector library
    def vectorizeDocument(self, infile):
        sentences = []
        with open(infile, 'r') as f:
            for line in f.readlines():
                sentence = line[2:].lower().strip(". \n")
                sentences.append(sentence.split(" "))
        # Create model to vectorize sentences based on training data
        document = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
        self.model = Doc2Vec(document, vector_size=100, window=2, min_count=1, workers=4)

        self.labels =[[], []]
        with open(infile, 'r') as f:
            for line in f.readlines():
                rating = int(line[0])
                sentence = line[2:].lower().strip(". \n").split(" ")

                self.labels[rating].append(self.model.infer_vector(sentence))
       
    def libraryVectorize(self, sentence): 
        return self.model.infer_vector(sentence)

    def featureVectorize(self, sentence):
        x = self.isHateSpeech(line[2:])
        y = self.countBannedWords(line[2:])

        return (x,y)
    
    def vectorizeBoW(self, sentence):
        vector = [0]*len(self.dict)
        for word in str(sentence).split():
            word = self.dict.get(word.lower())
            if word:
                vector[word] = 1

        return vector 
    
    def predictAndComputeAccuracy(self, infile, type):
        predictedClassification = []
        actualClassification = []

        points = []
        for point in self.labels[0]:
            points.append((point, 0))
        for point in self.labels[1]:
            points.append((point, 1))

        with open(infile, 'r') as f:
            for line in f.readlines():
                actualRating = line[0]
                sentence = line[2:].strip("\n")

                actualClassification.append((sentence, int(actualRating)))
                
                sentenceVector = None
                if type == "features":
                    sentenceVector = self.featureVectorize(sentence)
                elif type == "library":
                    sentenceVector = self.libraryVectorize(sentence)
                else:
                    sentenceVector = self.vectorizeBoW(sentence)

                # Take the first k distances
                kDistances = sorted([(np.linalg.norm(np.subtract(point, (sentenceVector))), rating) \
                    for (point, rating) in points], key=lambda x: x[0])[:self.K]
                
                genderedLikelihood = sum([n for _, n in kDistances])/float(self.K)

                # TODO Break ties by distances
                if genderedLikelihood <= 0.5:
                    predictedClassification.append((sentence, 0))
                    ## For graphing purposes
                    if type == "features":
                        self.labelsTest[0].append((x,y))
                else:
                    predictedClassification.append((sentence, 1))
                    if type == "features":
                        self.labelsTest[1].append((x,y))
            
        correct = 0
        for i in range(len(predictedClassification)):
            # Check if each sentence was given the same rating
            if predictedClassification[i][1] == actualClassification[i][1]:
                correct += 1

        accuracy = correct/float(len(predictedClassification))
        return (accuracy)
    
    def graphScatter(self, labels, dataType):
        scale = 200.0 * np.random.rand(len(labels[0]+labels[1]))

        for index, label in enumerate(labels):
            x, y = zip(*label)
            color, label = "black", "n/a"

            if index == 0: # Non-gendered
                color, label = "blue", "valid"
            else: # gendered
                color, label= "red", "invalid"
            try:
                plt.scatter(x, y, alpha=0.5, s=scale, c=color, label=label)
            except:
                continue

        plt.title('K-NN: {} Data, K={}'.format(dataType, K))
        plt.xlabel = "Hatespeech Score"
        plt.ylabel = "Number of banned words"
        plt.legend()
        plt.show()     
                

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

    def fitModelNaiveBayes(self, k=2):
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
                p = float(k + self.featureFreq[rating][feature_i]) \
                    / (self.totalSentencesForFeature(featureName, rating) + \
                     k*self.totalOutcomesForFeature(featureName, rating)) #P(Fi|Y)
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

    def tuneK(self, infile):
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
        accuracies = range(0, 100)

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

    def fitModelNaiveBayes(self, k=10):
        """
        Now you'll fit the model. For historical reasons, we'll call it F.
        F[rating][word] is -log(p(word|rating)).

        P(word|rating) =
        (alpha + self.counts[rating][word]) /
          (sum(self.counts[rating]) + (alpha * len(self.counts[rating])))


        ### LAPLACE SMOOTHING ### Smooth each conditionaing indepedently:
        P(word|rating) = count(word, rating) + k /
            count(rating) + k|number of words|
        """
        self.F = [[0] * len(self.dict) for _ in range(2)]

        for rating in range(2):
            for word in self.dict:
                word_index = self.dict[word]
                p = (self.counts[rating][word_index] + k) \
                    / float(sum(self.counts[rating]) +k*len(self.counts[rating]))
                        #P(Wi|Y)
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

    def tuneK(self, infile):
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
        accuracies = range(0, 1000)

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
    # print ("Laplace Smmoothing good k: ", c.tuneK('mini.valid'))
    # print ("TIME:", (bagOfWordsEnd - bagOfWordsStart))
    # print ("MEMORY:", (asizeof.asizeof(c)))

    # print ("FEATURE NAIVE BAYES CLASSIFIER")
    # featuresStart = time.time()
    # c = OtherFeaturesClassifier()
    # print ("Processing training set...")
    # c.buildModel('mini.train')
    # print ("Fitting model...")
    # c.fitModelNaiveBayes()
    # print ("Accuracy on validation set:", c.predictAndFindAccuracy('mini.valid')[1])
    # featuresEnd = time.time()
    # print ("Laplace Smoothing good k: ", c.tuneK('mini.valid'))
    # print ("TIME:", (featuresEnd - featuresStart))
    # print ("MEMORY:", (asizeof.asizeof(c)))

    # import matplotlib.pyplot as plt
    # print ("K-Nearest Neighbors Classification")
    # K = 2

    # featuresStart = time.time()
    # c = knnClassification(K)
    # ### Training ###
    # print ("Processing training set...")

    # trainingStart = time.time()

    # c.loadTrainingDataUsingFeatures('mini.train')

    # trainingEnd = time.time()
    # trainingTime = trainingEnd-trainingStart
    # trainingSize = asizeof.asizeof(c)
    # c.graphScatter(c.labels, "Training")

    # ### Testing ###
    # print ("Processing test set...")

    # testingStart = time.time()
    # accuracy = c.predictAndComputeAccuracy('mini.valid', "features")
    # testingEnd = time.time()
    # testingTime = testingEnd - testingStart
    # testingSize = asizeof.asizeof(c)

    # ### Graph Labeling & Formatting ###
    # c.graphScatter(c.labels, "Test") # Labels will now include test data

    # print("Accuracy: ", accuracy)
    # print("Time on Training: ", trainingTime)
    # print("Time on Testing: ", testingTime)
    # print("Training Memory: ", trainingSize)
    # print("Testing Memory: ", testingSize)

################Vectors########################

    print ("K-Nearest Neighbors Classification")
    K = 2

    c = knnClassification(K)
    print("processing training")
    trainingStart = time.time()

    c.vectorizeDocument("mini.train")
    trainingEnd = time.time()
    print("processing test")

    testingStart = time.time()
    accuracy = c.predictAndComputeAccuracy('mini.valid', "library")

    testingEnd = time.time()
    testingTime = testingEnd - testingStart
    trainingTime = trainingEnd-trainingStart
    trainingSize = asizeof.asizeof(c)

    testingSize = asizeof.asizeof(c)
    print("Accuracy: ", accuracy)

    print("Time on Training: ", trainingTime)
    print("Time on Testing: ", testingTime)
    print("Training Memory: ", trainingSize)
    print("Testing Memory: ", testingSize)
    # featuresStart = time.time()
    # c = knnClassification(K)
    # ### Training ###
    # print ("Processing training set...")

    # trainingStart = time.time()

    # c.loadTrainingDataUsingVectors('mini.train')

    # trainingEnd = time.time()
    # trainingTime = trainingEnd-trainingStart
    # trainingSize = asizeof.asizeof(c)
    # # c.graphScatter(c.labels, "Training")

    # ### Testing ###
    # print ("Processing test set...")

    # testingStart = time.time()
    # accuracy = c.predictAndComputeAccuracy('mini.valid', "vectors")
    # testingEnd = time.time()
    # testingTime = testingEnd - testingStart
    # testingSize = asizeof.asizeof(c)

    # ### Graph Labeling & Formatting ###
    # # c.graphScatter(c.labels, "Test") # Labels will now include test data

    # print("Accuracy: ", accuracy)
    # print("Time on Training: ", trainingTime)
    # print("Time on Testing: ", testingTime)
    # print("Training Memory: ", trainingSize)
    # print("Testing Memory: ", testingSize)

    # c = knnClassification(2)
    # c.loadTrainingDataUsingVectors("mini.train")




    

