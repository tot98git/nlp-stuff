from classifier import Classifier
from word2vec import WordEmbedding
from bigrams import Bigrams 

def main():
    nb = Classifier()
    w2v = WordEmbedding()

    nb.main()
    w2v.main()

if __name__=="__main__": 
    main() 