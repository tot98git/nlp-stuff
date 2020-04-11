import io
import re    # for preprocessing
import pandas as pd    #For data handling 
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec, KeyedVectors
from spacy.lang.sq import Albanian
import spacy    #For preprocessing

# Given the same set of data, we build word vectors which we can use to discover the proximity between words, among other things.

class WordEmbedding:
    @staticmethod
    def readFile(): 
        nlp = Albanian() # Albanian model doesn't support much things compared to other Spacy's models. It only supports stopwords.
        data = pd.read_csv("data/data.csv");
        formatted_titles = data.titulli.map(lambda text: text.replace('ë', 'e')).to_string() # Converted ë to e-s because Spacy's albanian stopwords source included stopwords written with e instead of ë
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        return nlp(formatted_titles)

    @staticmethod
    def clean(doc):
        # Lemmatizes and removes stopwords
        # Ideally we would extract other informations such as lemmas, word tags, dependency but Spacy's albanian package currently doesn't support them. 
        # The most we can do is extract the word (text). We extract each text item unless it is a stop word or a punctuation.
        txt = [token.text.lower() for token in doc if not (token.is_stop or token.is_punct)]

        # Word2Vec uses context words to learn the vector representation of a target word,
        # if a sentence is only one or two words long,
        # the benefit for the training is very small
        if len(txt) > 2:
            return ' '.join(txt)

    def getSentences(self):
        data = self.readFile()
        return [self.clean(sent).split() for sent in data.sents]

    @staticmethod
    def buildBigrams(txt):
        # Now we build bigrams. The benefits in our case are minimal because of the lacking support.

        phrases = Phrases(txt, min_count=30, progress_per=10000)
        bigram = Phraser(phrases)
        sentences = bigram[txt]
        return sentences

    @staticmethod
    def trainModel(sentences):
        cores = multiprocessing.cpu_count()   #Count the number of cores in a computer
        w2v_model = Word2Vec(min_count=1,
                            window=2,
                            size=300,
                            sample=6e-5, 
                            alpha=0.03, 
                            min_alpha=0.0007, 
                            negative=20,
                            workers=cores-1)
        w2v_model.build_vocab(sentences, progress_per=10000)
        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

        return w2v_model

    def main(self):
        sentences = self.getSentences()
        bigrams = self.buildBigrams(sentences)
        model = self.trainModel(bigrams)
        print(model.wv.most_similar(positive=["albin"]))
