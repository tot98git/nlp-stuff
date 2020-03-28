import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class Classifier:
    def readCsv(self):
        lines = pd.read_csv("data/data.csv");
        x = lines.titulli
        y = lines.kategoria
        return x, y
    
    def train_test_split(self):
        x, y = self.readCsv()
        return train_test_split(x, y, test_size=0.3, random_state=1) 


    def main(self):
        X_train, X_test, y_train, y_test = self.train_test_split()
        cv = CountVectorizer()
        training_data = cv.fit_transform(X_train)
        naive_bayes = MultinomialNB()
        naive_bayes.fit(training_data, y_train)
        testing_data = cv.transform(["Albin Kurti"])

        predictions = naive_bayes.predict(testing_data)
        print(predictions)
