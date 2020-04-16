from collections import defaultdict, Counter
from nltk import bigrams 
import pandas as pd
import dill as pickle 
from time import time    #To time our operations


class Bigrams:
    @staticmethod
    def readCsv(size = 10000):
        lines = pd.read_csv("data/simpsons_dataset.csv", nrows = size);
        x = lines.spoken_words
        return x.dropna()

    def build_bigram_model(self, size):
        bigram_model = defaultdict(lambda: defaultdict(lambda: 0))
        data = self.readCsv(size)
        
        print('Build started')

        t = time()
        
        for index, sentence in enumerate(data):
            print('Building bigrams for sentence', index + 1)

            try: 
                sentence = [word.lower() for word in sentence.split()]

                for w1, w2 in bigrams(sentence):
                    bigram_model[w1][w2] += 1

                for w1 in bigram_model:
                    total_count = float(sum(bigram_model[w1].values()))
                    
                for w2 in bigram_model[w1]:
                    bigram_model[w1][w2] /= total_count
            except: 
                print('Error at sentence: ', sentence)

        print('Build finished at {} mins.'.format(round((time() - t) / 60, 2)))

        return bigram_model

    @staticmethod
    def predict_next_word( model, first_word):
        second_word = model[first_word.lower()]

        top10words = Counter(second_word).most_common(10)

        predicted_words = list(zip(*top10words))[0]
        probability_score = list(zip(*top10words))[1]

        print(predicted_words, probability_score)

    def saveModel(self):
        # USE THIS METHOD IF THIS IS THE FIRST TIME YOU ARE USING THIS SCRIPT OR WANT TO REBUILD AND SAVE THE MODEL

        model = self.build_bigram_model(30000)
        saved_model = open('data/model', 'wb')
        pickle.dump(model, saved_model)
        return model

    @staticmethod
    def readModel():
        # USE THIS METHOD IF THE MODEL IS ALREADY BUILT AND WANT TO AVOID HAVING TO RUN THE BUILD METHOD FOR EVERY PREDICTION
        
        return pickle.load(open('data/model', 'rb'))

    def main(self):
        try: 
            # model = self.saveModel()
            model = self.readModel()
            self.predict_next_word(model, 'hey')
        except:
            print('Error reading the model')

if __name__=="__main__": 
    Bigrams().main() 
        