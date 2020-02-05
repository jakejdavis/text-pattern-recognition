import string
import ujson
from tqdm import tqdm
from operator import itemgetter
import os
from collections import Counter

import classes

from typing import List, Dict, Any, Union


def generate_ngrams(words: str, n: int):
    words_list: List[str] = words.split(' ')
    return [
        ' '.join(words_list[i:i + n])
        for i in range(len(words_list) - n + 1)
    ]


class NGramTrainer(classes.Trainer):
    def __init__(self, n, filename):
        super().__init__()

        self.n: int = n
        self.ngrams: List[str] = []
        self.filename: str = filename
        self.probabilities: Dict[str, Dict[str, float]] = {}

    def calculate_probabilities(self) -> Dict[str, Dict[str, float]]:
        totals: Dict[str, Dict[str, int]] = {}

        # Iterate over all the ngrams, excluding last element (hence [:-1])
        # as it does not have a 'next word'
        for i, ngram in enumerate(tqdm(self.ngrams[:-1])):
            # Get the second word from the next ngram (first word is the same)
            next_word: str = self.ngrams[i + 1].split(' ')[-1]
            if ngram not in totals:
                totals[ngram] = {}
            if next_word not in totals[ngram]:
                totals[ngram][next_word] = 1
            else:
                totals[ngram][next_word] += 1

        probabilities: Dict[str, Dict[str, float]] = {}
        for total_key, total_dict in tqdm(totals.items()):
            total_total = sum(total_dict.values())
            probabilities[total_key] = {key: value / total_total for key, value in total_dict.items()}
        return probabilities

    def train(self, filename: str):
        # Remove punctuation from corpora
        corpora: str = (open(filename, 'r', encoding="utf-8").read().translate(
            str.maketrans('', '', string.punctuation))
        ).lower()

        print(f"Loaded {len(corpora)} characters")

        print("Generating n-grams...")
        self.ngrams = generate_ngrams(corpora, self.n)
        print(f"Generated {len(self.ngrams)} n-grams")

        print("Calculating probabilities...")
        self.probabilities = self.calculate_probabilities()

        print("Writing probabilities...")
        self.write_probabilities()

    def load_probabilities(self) -> bool:
        if os.path.isfile(self.filename):
            with open(self.filename, 'r') as f:
                print("Decoding json...")
                self.probabilities = ujson.load(f)
                return True
        return False

    def write_probabilities(self):
        def merge_probabilities(probabilities_1, probabilities_2):
            merged_probabilities: Dict[Any, Union[Counter[Any], Any]] = {}
            print("Probabilities 1 pass...")
            for key, value in tqdm(probabilities_1.items()):
                if key not in probabilities_2:
                    merged_probabilities[key] = value
                else:
                    probabilities_1_value = {k: v / 2 for k, v in value.items()}
                    probabilities_2_value = {k: v / 2 for k, v in probabilities_2[key].items()}

                    # Use Counter class to merge dictionaries
                    merged_probabilities[key] = Counter(probabilities_1_value) + Counter(probabilities_2_value)
            print("Probabilities 2 pass...")
            for key, value in tqdm(probabilities_2.items()):
                if probabilities_2[key] is None:
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
        super().__init__()

        self.n: int = n
        self.trainer: NGramTrainer = NGramTrainer(n, f"ngram/probabilities_n{str(n)}.json")

    def init(self):
        if not self.trainer.load_probabilities():
            return False

    def train(self, filename: str):
        self.trainer.train(filename)

    def trial(self, filename):
        while True:
            to_predict: str = input("Text: ")
            if to_predict in self.trainer.probabilities:
                predicted_probabilities: Dict[str, float] = self.trainer.probabilities[to_predict]

                for predicted_word, predicted_probability in sorted(predicted_probabilities.items(), key=itemgetter(1),
                                                                    reverse=True)[0:5]:
                    print(f"Predicted '{predicted_word}' with {predicted_probability * 100}% confidence")

    def test(self, corpora):
        ngrams: List[str] = generate_ngrams(corpora, self.n)

        correct: int = 0
        incorrect: int = 0

        for i, test_trial in tqdm(enumerate(ngrams[0:len(ngrams) - 2])):
            next_word: str = ngrams[i + 1].split(' ')[-1]
            if test_trial in self.trainer.probabilities:
                predicted_probabilities = self.trainer.probabilities[test_trial]

                predicted_word: str = ""
                predicted_probability: float = 0
                predicted_word, predicted_probability = sorted(
                    predicted_probabilities.items(), key=itemgetter(1), reverse=True
                )[0]

                if predicted_word == next_word:
                    correct += 1
                else:
                    incorrect += 1

        return correct, incorrect
