from typing import Dict, List, Tuple
from gensim.models import KeyedVectors
import numpy as np
from operator import itemgetter
from tqdm import tqdm
from sner import POSClient
import string

import classes
import tag_predictor
import word_predictor
import ngram


class LSTMLSTMImplementation(classes.Implementation):

    def __init__(self, n, corpora_path):
        super().__init__()

        self.n = n

        wp_prefix: str = f"word_predictor/{n}/"
        word_predictor_filenames: Dict[str, str] = {
            "dataset": wp_prefix + "dataset.json",
            "word_vectors": corpora_path + ".wv",
            "model_json": wp_prefix + "model_json.json",
            "model_weights": wp_prefix + "model_weights.h5"
        }

        self.word_predictor_trainer: word_predictor.WordPredictorTrainer = \
            word_predictor.WordPredictorTrainer(n, word_predictor_filenames, "mse", "adam")

        tp_prefix = f"tag_predictor/{n}/"
        tag_predictor_filenames = {
            "dataset": tp_prefix + "dataset.json",
            "one_hot_labels": tp_prefix + "one_hot_labels.json",
            "model_json": tp_prefix + "model_json.json",
            "model_weights": tp_prefix + "model_weights.h5"
        }

        self.tag_predictor_trainer: tag_predictor.TagPredictorTrainer = \
            tag_predictor.TagPredictorTrainer(n, tag_predictor_filenames, "categorical_crossentropy", "adam")

    def init(self, corpora):
        self.word_predictor_trainer.init_model()
        self.tag_predictor_trainer.init_model()

    def predict(self, text, tagger, upos_1h_labels, word_vectors, predicting: int = 10):
        upos_texts = [(word[0], word[1]) for word in tagger.tag(text)]
        words = text.split(" ")
        if len(upos_texts) == 3 and len(words) == 3:
            # TAG PREDICTION
            X_upos: List[List[int]] = [list(upos_1h_labels.values())[
                          tag_predictor.UPOS_TAGS.index(upos)
                      ] for upos in [item[1] for item in upos_texts]]  # One-hot encode input UPOS tags
            prediction_upos: List[int] = self.tag_predictor_trainer.model.predict(np.array([
                X_upos
            ], dtype=np.float32))[0]  # Generate prediction of following UPOS tag

            # WORD PREDICTION
            X_words_vectors: List[List[int]] = [
                word_vectors[x_i] for x_i in words
            ]  # Word vector encode input text
            # Generate prediction of following word vector
            prediction_word_vector: List[int] = self.word_predictor_trainer.model.predict(np.array([X_words_vectors]))[0]
            # Use cosine similarity to determine most similar words from word vector
            predicted_words: List[List[str, float]] = word_vectors.most_similar(
                positive=[prediction_word_vector], topn=predicting)

            # predicted_word[0]: word, predicted_word[1]: confidence
            possible_sentences: List[str] = [text + " " + predicted_word[0] for predicted_word in predicted_words]
            possible_sentences_doc: List[Tuple[str, str]] = [
                tagger.tag(possible_sentence) for possible_sentence in possible_sentences
            ]

            combined_predictions = {}
            for i, possible_sentence_doc in enumerate(possible_sentences_doc):
                possible_word_upos: str = possible_sentence_doc[-1][1]
                possible_word: str = possible_sentence_doc[-1][0]

                upos_confidence: float = prediction_upos[tag_predictor.UPOS_TAGS.index(possible_word_upos)]

                combined_predictions[possible_word]: float = predicted_words[i][1] * upos_confidence
            return combined_predictions

    def train(self, filename):
        self.word_predictor_trainer.train(filename)
        self.tag_predictor_trainer.train(filename)

    def trial(self, filename):
        tagger = POSClient(host='localhost', port=9198)
        upos_1h_labels = tag_predictor.get_upos_1h_labels()

        word_vectors = KeyedVectors.load(filename + ".wv", mmap='r')

        while True:
            to_predict = input("Text: ")
            combined_predictions = self.predict(to_predict, tagger, upos_1h_labels, word_vectors)

            if combined_predictions:
                for predicted_word, predicted_probability in sorted(combined_predictions.items(), key=itemgetter(1),
                                                                    reverse=True):
                    print(f"Predicted '{predicted_word}' with {predicted_probability * 100}% confidence")

    def test(self, filename):
        # Remove punctuation from corpora
        corpora: str = (open(filename, 'r', encoding="utf-8").read().translate(
            str.maketrans('', '', string.punctuation))
        ).lower()

        tagger = POSClient(host='localhost', port=9198)
        upos_1h_labels = tag_predictor.get_upos_1h_labels()

        ngrams = ngram.generate_ngrams(corpora, self.n)[0:1999]

        word_vectors = KeyedVectors.load(filename + ".wv", mmap='r')

        correct: int = 0
        incorrect: int = 0

        for i, test_trial in enumerate(tqdm(ngrams[0:len(ngrams) - 2])):
            next_word = ngrams[i + 1].split(' ')[-1]

            try:
                combined_predictions = self.predict(test_trial, tagger, upos_1h_labels, word_vectors, predicting=1)
                if combined_predictions:
                    predicted_word = max(combined_predictions, key=combined_predictions.get)
        
                    if predicted_word == next_word:
                        correct += 1
                    else:
                        incorrect += 1
            except:
                pass # word not in vocabulary
