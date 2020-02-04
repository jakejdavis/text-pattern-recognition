import numpy as np
from operator import itemgetter
from tqdm import tqdm
from sner import POSClient

import classes
import ngram

import tag_predictor


class LSTMNgramImplementation(classes.Implementation):
    def __init__(self, n):
        super().__init__()

        self.n = n

        prefix = f"tag_predictor/{n}/"
        tag_predictor_filenames = {
            "dataset": prefix + "dataset.json",
            "one_hot_labels": prefix + "one_hot_labels.p",
            "model_json": prefix + "model_json.json",
            "model_weights": prefix + "model_weights.h5"
        }

        self.tag_predictor_trainer = tag_predictor.TagPredictorTrainer(n, tag_predictor_filenames,
                                                                       "categorical_crossentropy", "adam")
        self.ngram_trainer = ngram.NGramTrainer(n, f"probabilities_n{str(n)}.json")

    def init(self, filename):
        if not self.ngram_trainer.load_probabilities():
            print(f"Training ngram {str(self.n)}")
            self.ngram_trainer.train(filename)

        self.tag_predictor_trainer.init()

    def predict(self, text, tagger, upos_1h_labels):
        upos_texts = [(word[0], word[1]) for word in tagger.tag(text)]
        if len(upos_texts) == 3:

            X_upos = [list(upos_1h_labels.values())[
                          tag_predictor.UPOS_TAGS.index(upos)
                      ] for upos in [item[1] for item in upos_texts]]
            prediction_upos = self.tag_predictor_trainer.model.predict(np.array([
                X_upos
            ], dtype=np.float32))[0]

            if text in self.ngram_trainer.probabilities:
                ngram_predictions = self.ngram_trainer.probabilities[text]
                possible_sentences = [text + ' ' + predicted_word for (predicted_word, predicted_probability) in
                                      ngram_predictions.items()]
                # Tokenise ngram possible sentences with stanfordnlp
                ngram_possible_sentences_doc = [tagger.tag(possible_sentence) for possible_sentence in
                                                possible_sentences]

                combined_predictions = {}
                for i, ngram_possible_sentence_doc in enumerate(ngram_possible_sentences_doc):
                    ngram_prediction_upos = ngram_possible_sentence_doc[-1][1]
                    ngram_prediction_word = ngram_possible_sentence_doc[-1][0]
                    try:
                        # Get confidence of upos
                        upos_confidence = prediction_upos[tag_predictor.UPOS_TAGS.index(ngram_prediction_upos)]

                        # Add combined prediction of ngram and upos prediction to combined prediction list
                        combined_predictions[ngram_prediction_word] = list(ngram_predictions.values())[
                                                                          i] * upos_confidence
                    except ValueError:
                        pass  # UPOS key not in UPOS Tags

                return combined_predictions
        return False

    def train(self, filename):
        self.tag_predictor_trainer.train(filename)

    def trial(self, filename):
        tagger = POSClient(host='localhost', port=9198)
        upos_1h_labels = tag_predictor.get_upos_1h_labels()

        while True:
            to_predict = input("Text: ")
            combined_predictions = self.predict(to_predict, tagger, upos_1h_labels)

            if combined_predictions:
                for predicted_word, predicted_probability in sorted(combined_predictions.items(), key=itemgetter(1),
                                                                    reverse=True)[0:5]:
                    print(f"Predicted '{predicted_word}' with {predicted_probability * 100}% confidence")

    def test(self, filename):
        corpora = open(filename, 'r').read()
        tagger = POSClient(host='localhost', port=9198)
        upos_1h_labels = tag_predictor.get_upos_1h_labels()

        ngrams = ngram.generate_ngrams(corpora, self.n)

        correct = 0
        incorrect = 0

        for i, test_trial in tqdm(enumerate(ngrams[0:len(ngrams) - 2])):
            next_word = ngrams[i + 1].split(' ')[-1]

            combined_predictions = self.predict(test_trial, tagger, upos_1h_labels)
            if combined_predictions:
                predicted_word, predicted_probability = \
                sorted(combined_predictions.items(), key=itemgetter(1), reverse=True)[0]

                if predicted_word == next_word:
                    correct += 1
                else:
                    incorrect += 1

        return correct, incorrect
