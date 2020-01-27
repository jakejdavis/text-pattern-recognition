import string
import ujson
from tqdm import tqdm
from operator import itemgetter
import os
from collections import Counter

import classes

from typing import List, Dict

CORPORA_PATH = "/Users/jakedavis/Documents/School/Computer Science/NEA/Predict/Production/corpora/en_US.blogs.txt"

class NGramTrainer(classes.Trainer):
    def __init__(self, n, filename):
        self.n = n
        self.ngrams = []
        self.filename = filename

    def generate_ngrams(self, words: str, n = -1):
        if n == -1: n = self.n

        words = words.split(' ')
        return [ 
            ' '.join(words[i:i+self.n])
            for i in range(len(words)-self.n+1)
        ]

    def calculate_probabilities(self):
        totals: Dict[str, Dict[str, int]] = {}

        # Iterate over all the ngrams, excluding last element (hence [:-1])
        # as it does not have a 'next word'
        for i, ngram in enumerate(tqdm(self.ngrams[:-1])):
            # Get the second word from the next ngram (first word is the same)
            next_word = self.ngrams[i+1].split(' ')[-1]
            if not ngram in totals:
                totals[ngram] = {}
            if not next_word in totals[ngram]:
                totals[ngram][next_word] = 1
            else:
                totals[ngram][next_word] += 1
        
        probabilities: Dict[str, Dict[str, int]] = {}
        for total_key, total_dict in tqdm(totals.items()):
            total_total = sum(total_dict.values())
            probabilities[total_key] = {key: value / total_total for key, value in total_dict.items()}
        return probabilities
    
    def train(self, filename: str):
        # Remove punctuation from corpora
        corpora = (open(filename, 'r', encoding="utf-8").translate(
            str.maketrans('', '', string.punctuation))
        ).lower()      

        print(f"Loaded {len(corpora)} characters")  

        print("Generating n-grams...")
        self.ngrams = self.generate_ngrams(corpora)
        print(f"Generated {len(self.ngrams)} n-grams")

        print("Calculating probabilities...")
        self.probabilities = self.calculate_probabilities()

        print("Writing probabilities...")
        self.write_probabilities()

    def load_probabilities(self):
        if os.path.isfile(self.filename):
            with open(self.filename, 'r') as f:
                print("Decoding json...")
                self.probabilities = ujson.load(f)
                return True
        return False

    def write_probabilities(self):
        def merge_probabilities(probabilities_1, probabilities_2):
            merged_probabilities = {}
            print("Probabilities 1 pass...")
            for key, value in tqdm(probabilities_1.items()):
                if not key in probabilities_2:
                    merged_probabilities[key] = value
                else:
                    probabilities_1_value = {k: v / 2 for k,v in value.items()} 

                    probabilities_2_value = {k: v / 2 for k,v in probabilities_2[key].items()}
                    
                    merged_probabilities[key] = Counter(probabilities_1_value) + Counter(probabilities_2_value)
            print("Probabilities 2 pass...")
            for key, value in tqdm(probabilities_2.items()):
                if probabilities_2[key] == None:
                    merged_probabilities[key] = value
            return merged_probabilities

        if os.path.isfile(self.filename):
            print("Merging probabilities...")
            probabilities_to_write = merge_probabilities(ujson.load(open(self.filename, 'r')), self.probabilities)
        else:
            probabilities_to_write = self.probabilities
        with open(self.filename, 'w') as f:
            f.write(ujson.dumps(probabilities_to_write))
        print(f"Wrote probabilities to {self.filename}")

class NGramImplementation(classes.Implementation):
    def __init__(self, n):
        self.n = n
        self.trainer = NGramTrainer(n, f"probabilities_n{str(n)}.json")

    def init(self):
        if not self.trainer.load_probabilities():
            return False
            #print(f"Training ngram {str(self.n)}")
            #self.trainer.train()

    def train(self, filenmame):
        self.trainer.train(filename)
    
    def trial(self):
        while True:
            to_predict = input("Text: ")
            if to_predict in self.trainer.probabilities:
                predicted_probabilities = self.trainer.probabilities[to_predict]

                for predicted_word, predicted_probability in sorted(predicted_probabilities.items(), key=itemgetter(1), reverse=True)[0:5]:
                    print(f"Predicted '{predicted_word}' with {predicted_probability*100}% confidence")

    def test(self, corpora):
        ngrams = self.trainer.generate_ngrams(corpora, self.n)

        correct = 0
        incorrect = 0

        for i, test_trial in tqdm(enumerate(ngrams[0:len(ngrams)-2])):
            next_word = ngrams[i+1].split(' ')[-1]
            if test_trial in self.trainer.probabilities:
                predicted_probabilities = self.trainer.probabilities[test_trial]

                predicted_word, predicted_probability = sorted(predicted_probabilities.items(), key=itemgetter(1), reverse=True)[0]
                if predicted_word == next_word:
                    correct += 1
                else:
                    incorrect += 1

        return (correct, incorrect)