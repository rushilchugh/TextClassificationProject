import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import word_tokenize
import re
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class SentenceClassification:

    def __init__(self, trainingLoc, delimiter = ","):
        self.trainingLoc = trainingLoc
        self.delimiter = delimiter
        self.uniqueDict = None
        self.df = None
        self.dfSize = None
        self.freqDist = None
        self.freqDistBiGram = None

    def createTrainingDataframe(self):

        """
        Different for different kinds of input
        """

        self.df = pd.read_csv(self.trainingLoc, sep=self.delimiter, names=['label', 'text'])
        self.df['label'] = self.df['label'].astype(bool)

    def vectorizeInput(self):
        #CountVecotrizer sans numerics
        cv = CountVectorizer(preprocessor = lambda X:   re.sub(r".*\d.*", "0NUM0", X.lower()))
        X = cv.fit_transform(self.df.text)
        cvDf = pd.DataFrame(X.todense(), columns = cv.get_feature_names())

        self.df = pd.concat([self.df, cvDf], axis = 1)

        self.dfSize = len(self.df)
        self.uniqueDict = cv.vocabulary_

    def featureEngineering(self):

        #Feature 1 - Sentence Length and Words
        self.df['sentenceLength'] = pd.Series([len(re.sub(r'[^\w\s]', '', i)) for i in self.df.text])

        #Feature 2 - Number of Words
        self.df['numberOfWords'] = pd.Series([len(word_tokenize(re.sub(r'[^\w\s]', '', i))) for i in self.df.text])

        #Feature 3 - Number of Capital
        self.df['Capitals'] = self.df['text'].apply(lambda comment: sum([1 for C in comment if C.isupper()]))

        #Feature 4 - Percentage of Capitals
        self.df['percentageCapitals'] = self.df.apply(lambda row:   float(row['Capitals']) / float(row['sentenceLength']), axis=1)

        #Feature 5 - Number of Exclamations
        self.df['numberOfExclamations'] = self.df['text'].apply(lambda x:   x.count('!'))

    def Visualizations(self):
        self.df.label.value_counts().plot.pie(explode=[0, 0.1], shadow=True)

    def dimensionalityReduction(self):
        pass

    def modelData(self):
        """
        Base Level Models
            1.  Random Forest Classifier
            2. Gradient Boosting Machine
            3.  BiGram Bag Of Words
            4. Support Vector Machine
            5.  K-Nearest Neighbors

        Second Level Models
            1.  XGBoost Classifier
            2. Logistic Regression
        """

    def Predictions(self):
        pass

te = SentenceClassification(trainingLoc = r'C:\NOS\Coding_6\Project\Datasets\Phase A\training.txt', delimiter = '\t')
te.createTrainingDataframe()
te.vectorizeInput()
te.featureEngineering()


