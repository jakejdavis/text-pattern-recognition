from typing import List, Tuple, Dict
import numpy as np
from operator import itemgetter
from tqdm import tqdm
from sner import POSClient
import string

import classes
import ngram
import tag_predictor


class LSTMNGramImplementation(classes.Implementation):
    def __init__(self, n):
        super().__init__()

        self.n: int = n

        prefix = f"tag_predictor/{n}/"
        tag_predictor_filenames = {
            "dataset": prefix + "dataset.json",
            "one_hot_labels": prefix + "one_hot_labels.p",
            "model_json": prefix + "model_json.json",
            "model_weights": prefix + "model_weights.h5"
        }

        self.tag_predictor_trainer = tag_predictor.TagPredictorTrainer(n, tag_predictor_filenames,
                                                                       "categorical_crossentropy", "adam")
        self.ngram_trainer = ngram.NGramTrainer(n, f"ngram/probabilities_n{str(n)}.json")

    def init(self, filename):
        if not self.ngram_trainer.load_probabilities():
            print(f"Training ngram {str(self.n)}")
            self.ngram_trainer.train(filename)

        self.tag_predictor_trainer.init_model()

    def predict(self, text, tagger, upos_1h_labels) -> Dict[str, float]:
        upos_texts = [(word[0], word[1]) for word in tagger.tag(text)]
        if len(upos_texts) == self.n:  # Ensure input data has correct dimensions
            X_upos: List[List[int]] = [list(upos_1h_labels.values())[
                                           tag_predictor.UPOS_TAGS.index(upos)
                                       ] for upos in [item[1] for item in upos_texts]]  # One-hot encode input UPOS tags
            prediction_upos: List[int] = self.tag_predictor_trainer.model.predict(np.array([
                X_upos
            ], dtype=np.float32))[0]  # Generate prediction of following UPOS tag

            if text in self.ngram_trainer.probabilities:  # Ensure context is in ngram trainer probabilities

                # Get next word probabilities using context as key
                ngram_predictions: Dict[str, float] = self.ngram_trainer.probabilities[text]

                # Create possible sentences adding potential word to input text
                possible_sentences: List[str] = [text + ' ' + predicted_word for (predicted_word, predicted_probability)
                                                 in ngram_predictions.items()]

                # Tokenise ngram possible sentences with sner tagger
                ngram_possible_sentences_doc: List[Tuple[str, str]] = [tagger.tag(possible_sentence) for
                                                                       possible_sentence in
                                                                       possible_sentences]

                combined_predictions = {}
                for i, ngram_possible_sentence_doc in enumerate(ngram_possible_sentences_doc):
                    ngram_prediction_upos: str = ngram_possible_sentence_doc[-1][1]
                    ngram_prediction_word: str = ngram_possible_sentence_doc[-1][0]
                    try:
                        # Get confidence of upos
                        upos_confidence: float = prediction_upos[tag_predictor.UPOS_TAGS.index(ngram_prediction_upos)]

                        # Add combined prediction of ngram and upos prediction to combined prediction list
                        combined_predictions[ngram_prediction_word]: float = list(ngram_predictions.values())[
                                                                                 i] * upos_confidence
                    except ValueError:
                        pass  # UPOS key not in UPOS Tags

                return combined_predictions
        return False

    def train(self, filename):
        self.tag_predictor_trainer.train(filename)

    def trial(self, filename):
        tagger = POSClient(host='localhost', port=9198)
        upos_1h_labels: Dict[str, List[int]] = tag_predictor.get_upos_1h_labels()

        while True:
            to_predict = input("Text: ")
            combined_predictions: Dict[str, float] = self.predict(to_predict, tagger, upos_1h_labels)

            if combined_predictions:
                for predicted_word, predicted_probability in sorted(combined_predictions.items(), key=itemgetter(1),
                                                                    reverse=True)[0:5]:
                    print(f"Predicted '{predicted_word}' with {predicted_probability * 100}% confidence")

    def test(self, filename):
        # Remove punctuation from corpora
        corpora: str = (open(filename, 'r', encoding="utf-8").read().translate(
            str.maketrans('', '', string.punctuation))
        ).lower()

        tagger = POSClient(host='localhost', port=9198)
        upos_1h_labels = tag_predictor.get_upos_1h_labels()

        ngrams = ngram.generate_ngrams(corpora, self.n)

        correct = 0
        incorrect = 0

        for i, test_trial in enumerate(tqdm(ngrams[0:len(ngrams) - 2])):
            next_word = ngrams[i + 1].split(' ')[-1]

            combined_predictions = self.predict(test_trial, tagger, upos_1h_labels)
            if combined_predictions:
                predicted_word = max(combined_predictions, key=combined_predictions.get)

                if predicted_word == next_word:
                    correct += 1
                else:
                    incorrect += 1

        return correct, incorrect
