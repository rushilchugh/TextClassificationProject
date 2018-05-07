import pandas as pd

import re

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from xgboost import XGBClassifier

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class SentenceClassification:

    def __init__(self, trainingLoc, delimiter = ",", stemmer = SnowballStemmer('english')):
        self.trainingLoc = trainingLoc
        self.delimiter = delimiter

        self.df = None
        self.predDf = None

        self.stemmer = stemmer
        self.cv = None

        self.MinMaxScaling = MinMaxScaler()

        self.clfRf = None
        self.clfSVC = None
        self.clfAdaBoost = None
        self.clfGb = None
        self.clfETC = None
        self.clfXGB = None

        self.feListRf = None
        self.feListSVC = None
        self.feListAdaBoost = None
        self.feListGB = None
        self.feListETC = None

    def createTrainingDataframe(self):

        """
        Different for different kinds of input
        """

        self.df = pd.read_csv(self.trainingLoc, sep=self.delimiter, names=['label', 'text'])
        self.df['label'] = self.df['label'].astype(bool)

        #Stem the Text Column
        te.df['stemText'] = te.df['text'].apply(lambda X: ' '.join([te.stemmer.stem(word) for word in word_tokenize(X)]))

    def vectorizeInput(self):

        #CountVecotrizer sans numerics
        cv = CountVectorizer(preprocessor = lambda X:   re.sub(r".*\d.*", "", X.lower()), min_df=5, stop_words=stopwords.words('english'))
        X = cv.fit_transform(self.df['stemText'])
        cvDf = pd.DataFrame(X.todense(), columns = cv.get_feature_names())

        self.df = pd.concat([self.df, cvDf], axis = 1)

        self.cv = cv

    def featureEngineering(self):

        #Feature 1 - Sentence Length and Words
        self.df['sentenceLength'] = pd.Series([len(re.sub(r'[^\w\s]', '', i)) for i in self.df.text])

        #Feature 2 - Number of Words
        self.df['numberOfWords'] = pd.Series([len(word_tokenize(re.sub(r'[^\w\s]', '', i))) for i in self.df.text])

        #Feature 3 - Number of Capitals
        self.df['Capitals'] = self.df['text'].apply(lambda comment: sum([1 for C in comment if C.isupper()]))

        #Feature 4 - Percentage of Capitals
        self.df['percentageCapitals'] = self.df.apply(lambda row:   float(row['Capitals']) / float(row['sentenceLength']), axis=1)

        #Feature 5 - Number of Exclamations
        self.df['Exclamations'] = self.df['text'].apply(lambda x:   x.count('!'))

        #Feature 6 - Percentage Exclamations
        self.df['percentageExclamations'] = self.df.apply(lambda row: float(row['Exclamations']) / float(row['sentenceLength']),
                                                      axis=1)
    def featureScaling(self):
        self.df[['sentenceLength', 'numberOfWords', 'Capitals', 'Exclamations']] = self.MinMaxScaling.fit_transform(
            self.df[['sentenceLength', 'numberOfWords', 'Capitals', 'Exclamations']])

    def modelData(self):
        """
        Base Level Models
            1.  Random Forest Classifier
            2. Gradient Boosting Machine
            3. AdaBoost
            4. Support Vector Machine
            5.  K-Nearest Neighbors
            6. Extra Trees Classifier

        Second Level Models
            1.  XGBoost Classifier
            2. Logistic Regression
        """

        """
        Level 1 Models
        """

        #TrainTestSplit
        X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(self.df.drop(['label', 'text', 'stemText'], axis=1),
                                                            self.df['label'], test_size=0.2)

        clfRf = RandomForestClassifier(verbose=True, n_estimators=1000, n_jobs=-1)
        clfAdaBoost = AdaBoostClassifier(n_estimators=1000, learning_rate=0.50)
        clfGb = GradientBoostingClassifier(n_estimators=500, max_depth=5, min_samples_leaf=2, verbose=True)
        clfSVC = SVC(kernel='linear', verbose=True)
        clfETC = ExtraTreesClassifier(verbose=True, n_estimators=1000, n_jobs=-1)

        clfRf.fit(X_train_base, y_train_base)
        clfAdaBoost.fit(X_train_base, y_train_base)
        clfGb.fit(X_train_base, y_train_base)
        clfSVC.fit(X_train_base, y_train_base)
        clfETC.fit(X_train_base, y_train_base)

        self.clfRf = clfRf
        self.clfAdaBoost = clfAdaBoost
        self.clfGb = clfGb
        self.clfSVC = clfSVC
        self.clfETC = clfETC

        self.feListRf = sorted(list(zip(self.df.drop(['label', 'text', 'stemText'], axis=1), self.clfRf.feature_importances_)), key=lambda x:  x[1], reverse=True)
        self.feListAdaBoost = sorted(list(zip(self.df.drop(['label', 'text', 'stemText'], axis=1), self.clfAdaBoost.feature_importances_)), key=lambda x:  x[1], reverse=True)
        self.feListGB = sorted(list(zip(self.df.drop(['label', 'text', 'stemText'], axis=1), self.clfGb.feature_importances_)), key=lambda x:  x[1], reverse=True)
        self.feListSVC = sorted(list(zip(self.df.drop(['label', 'text', 'stemText'], axis=1), *self.clfSVC.coef_)), key=lambda x:  x[1], reverse=True)
        self.feListETC = sorted(list(zip(self.df.drop(['label', 'text', 'stemText'], axis=1), self.clfETC.feature_importances_)), key=lambda x:  x[1], reverse=True)

        """
        Generating Level 1 Predictions
        """

        self.predDf = pd.DataFrame({'RandomForests': clfRf.predict(X_test_base), 'AdaBoost': clfAdaBoost.predict(X_test_base),
                                    'GradientBoosting': clfGb.predict(X_test_base), 'ExtraTrees': clfETC.predict(X_test_base),
                                    'SupportVectorMachine': clfSVC.predict(X_test_base), 'ActualLabels': y_test_base})

        """
        Level 2 Models
        """
        X_train, X_test, y_train, y_test = train_test_split(self.predDf.drop(['ActualLabels'], axis=1),
                                                            self.predDf['ActualLabels'], test_size=0.2)

        clfXGBoost = XGBClassifier(n_estimators=2000, max_depth=5, min_child_weight= 2, gamma=0.9, subsample=0.8,
                                   colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1,
                                   silent=False)

        clfXGBoost.fit(X_train, y_train)

        self.clfXGB = clfXGBoost


    def Visualizations(self):

        fig, axes = plt.subplots(3, 3)

        axes[0, 0].bar(*zip(*self.feListGB[:10]))
        axes[0, 0].set_title('GBM Feature Importance')

        axes[0, 1].bar(*zip(*self.feListETC[:10]))
        axes[0, 1].set_title('ETC Feature Importance')

        axes[0, 2].bar(*zip(*self.feListAdaBoost[:10]))
        axes[0, 2].set_title('AdaBoost Feature Importance')

        axes[1, 0].bar(*zip(*self.feListRf[:10]))
        axes[1, 0].set_title('Rf Feature Importance')

        axes[1, 1].bar(*zip(*(self.feListSVC[:5] + self.feListSVC[-5:])))
        axes[1, 1].set_title('SVC Feature Importance')

        plt.subplots_adjust(hspace=0.8)
        #plt.tight_layout()

        for subPlot in axes.flat:
            for textLabel in subPlot.get_xticklabels():
                textLabel.set_rotation(70)

        plt.show()

    def featureImportanceList(self):

        #Random Forest Classifers
        print('------------------Random Forests Feature Importance List------------------\n')

        for feIM in self.feListRf:
            print(feIM)

        #SVM
        print('------------------Support Vector Machine(SVM) Feature Importance List------------------\n')

        for feIM in self.feListSVC:
            print(feIM)

        #AdaBoost
        print('------------------AdaBoost Feature Importance List------------------\n')

        for feIM in self.feListAdaBoost:
            print(feIM)

        #Gradient Boosting
        print('------------------Gradient Boosting Machine Feature Importance List------------------\n')

        for feIM in self.feListGB:
            print(feIM)


        #Extra Trees
        print('------------------Extra Trees Feature Importance List------------------\n')

        for feIM in self.feListETC:
            print(feIM)

    def predictSentence(self, sentence):
        """
        1. Tokenize
        2. Stemming
        3. Join Back Again
        4. Vectorize
            i. Remove Numbers
            ii. Remove Stopwords
        5. Make a dense array
        5. Create Features
            i. Sentence Length
            ii. Word Number
            iii. Capital Percentage
            iv. Number of Exclamations
            v. Exclamation Percentage
        6. Scale Features
        7. Predict category through First Level Models and create new features
        8. Predict final category through the Second Level Model trained upon the generated predictions
        """

te = SentenceClassification(trainingLoc=r'C:\NOS\Coding_6\Project\Datasets\Phase A\training.txt', delimiter='\t')
te.createTrainingDataframe()
te.vectorizeInput()
te.featureEngineering()
te.featureScaling()

print('Model Training Beginning')
te.modelData()

print('Model Training Done')
te.Visualizations()