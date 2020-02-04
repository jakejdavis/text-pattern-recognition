import argparse


import ngram
import lstm_ngram
import lstm_pure
from classes import Implementation


parser = argparse.ArgumentParser()
parser.add_argument('--implementation', help='Implementation')
parser.add_argument('--n', help='N')
parser.add_argument('--procedure', help='Procedure')
parser.add_argument('--corpora', help='Corpora')



TRAIN_CORPORA_PATH = "corpora/corpora_aa.txt"
TEST_CORPORA_PATH = "corpora/corpora_ab.txt"

def main(): 
    # Collect user input
    args = parser.parse_args()

    option = args.implementation if args.implementation else input("ngram/ngram with lstm/lstm with lstm: ")
    ngram_n_input = int(args.n) if args.n else int(input("ngram n: "))
    procedure = args.procedure if args.procedure else input("trial/test/train: ")
    corpora_path = args.corpora if args.corpora else input("corpora path: ")

    
    # Instantiate the respective class from user input
    if option == "ngram":
        # Pass the value of ngram_n_input as a class parameter
        implementation: Implementation = ngram.NGramImplementation(n = ngram_n_input)
    elif option == "ngram with lstm":
        implementation: Implementation = lstm_ngram.LSTMNgramImplementation(n = ngram_n_input)
    elif option == "lstm with lstm":
        implementation: Implementation = lstm_pure.LSTMLSTMImplementation(n = ngram_n_input, corpora_path = corpora_path)

    # Run init function for implementations with ANNs
    if option != "ngram":
        implementation.init(corpora_path)

    # Run the respective procedure from user input
    if procedure == "train":
        implementation.train(corpora_path)
    elif procedure == "trial":
        implementation.trial(corpora_path)
    elif procedure == "test":
        correct, incorrect = implementation.test(corpora_path)
        print(f"Implementation had {(correct/(correct+incorrect))*100}% accuracy on {correct+incorrect} iterations ({correct} correct, {incorrect} incorrect)")

if __name__ == "__main__":
    main()
