import string
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import get_tmpfile
import os
import ujson
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input, Dense, LSTM, Activation, Embedding
from tensorflow.keras.utils import Sequence
import stanfordnlp
from operator import itemgetter
from sner import POSClient

import classes
import tag_predictor

class WordPredictorDataGenerator(Sequence):
    def __init__(self, words_filename, kv_filename, n, batch_size=32):
        self.batch_size = batch_size
        with open(words_filename, 'r') as f:
            json = ujson.load(f)
            self.texts_length = 50000#json["len"]
            self.texts = json["texts"][0:49999]
        self.word_vectors = KeyedVectors.load(kv_filename, mmap='r')

        self.index = 0
        self.n = n

    def __len__(self):
        return int(np.floor(self.texts_length/3 / self.batch_size))

    def __getitem__(self, index):
        # Generates batch
        X = np.zeros([self.batch_size, self.n, self.word_vectors.vector_size])
        Y = np.zeros([self.batch_size, self.word_vectors.vector_size])

        i = 0
        # Generate X Y data until full (last item is not zero'd)
        while np.count_nonzero(X[self.batch_size-1]) == 0:
            x = None
            y = None
            try:
                x, y = self.__data_generation()
                X[i] = x
                Y[i] = y
                i += 1
            except Exception as e:
                print(e)
            self.index += 1
        return X, Y

    def __data_generation(self):
        x = np.array([
            self.word_vectors[v] for v in self.texts[self.index:self.index+self.n]
        ])
        y = np.array([self.word_vectors[self.texts[self.index+self.n]]])

        return x, y

class WordPredictorTrainer(classes.TrainerML):
    def __init__(self, n, filenames, loss, optimizer):
        # filenames keys: word_vectors

        self.n = n

        self.filenames = filenames

        self.loss = loss
        self.optimizer = optimizer

        self.X = []
        self.Y = []

        self.word_vectors = None

    def generate_dataset(self, corpora):
        print("Training Word2Vec model...")
        model = Word2Vec([corpora], size=100, window=5, min_count=1, workers=4)
        self.word_vectors = model.wv
        
        print("Saving word vectors")
        os.makedirs(os.path.dirname(self.filenames["word_vectors"]), exist_ok=True)
        self.word_vectors.save(self.filenames["word_vectors"])

        print("Generating X Y dataset")
        for i, text in enumerate(corpora[:-(self.n+1)]):
            self.X.append(corpora[i:i+self.n])
            self.Y.append(corpora[i+self.n])


    def encode_x_y(self):
        self.X = np.array([
            (self.word_vectors[text] for text in x_i) for x_i in self.X
        ])
        self.Y = np.array([self.word_vectors[y_i] for y_i in self.Y])

    def create_model(self):
        word_vectors = KeyedVectors.load(self.filenames["word_vectors"], mmap='r')
        self.model = Sequential()
        #self.model.add(Embedding(vocab_size, 50, input_length=self.X.shape[1]))
        self.model.add(LSTM(100, input_shape=(3, word_vectors.vector_size), return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(word_vectors.vector_size, activation='tanh'))

    def init(self, corpora):
        # Ensures self.X, self.Y and self.upos_dict are set
        """if not self.check_dataset():
            print("Dataset not initialized")
            print("Loading dataset...")
            if not self.load_dataset():
                print("Dataset load failed... preprocessing instead")
                self.preprocess(corpora, split = True)
                self.write_dataset([self.X.tolist(), self.Y.tolist()])"""
        
        if not self.load_model():
            print("Loading model failed... creating model instead")
            self.create_model()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        print("Model compiled!")

    def check_dataset(self):
        return self.X != [] and self.Y != []

    def load_dataset(self):
        if os.path.isfile(self.filenames["dataset"]):
            print("Loading dataset items...")
            with open(self.filenames["dataset"], 'r') as f:
                print("Decoding json...")
                json_list = ujson.load(f)
                self.X, self.Y = np.array(json_list[0]), np.array(json_list[1])

            self.word_vectors = KeyedVectors.load(self.filenames["word_vectors"], mmap='r')
            return True
        return False

    def train(self, filename):
        print("Training model...")
        training_generator = WordPredictorDataGenerator(filename+".uts", filename+".wv", self.n)

        callbacks_list = [classes.ModelSaver(self.filenames["model_weights"])]

        self.model.fit_generator(generator=training_generator, callbacks=callbacks_list)

        print("Writing model to {}".format(self.filenames["model_json"]))
        self.write_model()

class LSTMLSTMImplementation(classes.Implementation):
    def __init__(self, n, corpora_path):
        self.n = n

        wp_prefix = f"word_predictor/{n}/"
        word_predictor_filenames = {
            "dataset": wp_prefix + "dataset.json",
            "word_vectors": corpora_path + ".wv",
            "model_json": wp_prefix + "model_json.json",
            "model_weights": wp_prefix + "model_weights.h5"
        }

        self.word_predictor_trainer = WordPredictorTrainer(n, word_predictor_filenames, "mse", "adam")

        tp_prefix = f"tag_predictor/{n}/"
        tag_predictor_filenames = {
            "dataset": tp_prefix + "dataset.json",
            "one_hot_labels": tp_prefix + "one_hot_labels.json",
            "model_json": tp_prefix + "model_json.json",
            "model_weights": tp_prefix + "model_weights.h5"
        }

        self.tag_predictor_trainer = tag_predictor.TagPredictorTrainer(n, tag_predictor_filenames, "categorical_crossentropy", "adam")
        
    def init(self, corpora):
        self.word_predictor_trainer.init(corpora)
        self.tag_predictor_trainer.init()

    def predict(self, text, tagger, upos_1h_labels, word_vectors):
        upos_texts = [(word[0], word[1]) for word in tagger.tag(text)]
        words = text.split(" ")
        if len(upos_texts) == 3 and len(words) == 3:
            # TAG PREDICTION
            X_upos = [list(upos_1h_labels.values())[
                        tag_predictor.UPOS_TAGS.index(upos)
                    ] for upos in [item[1] for item in upos_texts]]
            prediction_upos = self.tag_predictor_trainer.model.predict(np.array([
                X_upos
            ], dtype=np.float32))[0]

            # WORD PREDICTION
            X_words_vectors = [
                word_vectors[x_i] for x_i in words
            ]
            prediction_word_vector = self.word_predictor_trainer.model.predict(np.array([X_words_vectors]))[0]

            predicted_words = word_vectors.most_similar(positive=[prediction_word_vector], topn=30)

            # predicted_word[0]: word, predicted_word[1]: confidence
            possible_sentences = [text + " " + predicted_word[0] for predicted_word in predicted_words]
            possible_sentences_doc = [tagger.tag(possible_sentence) for possible_sentence in possible_sentences]

            combined_predictions = {}
            for i, possible_sentence_doc in enumerate(possible_sentences_doc):
                possible_word_upos = possible_sentence_doc[-1][1]
                possible_word = possible_sentence_doc[-1][0]

                upos_confidence = prediction_upos[tag_predictor.UPOS_TAGS.index(possible_word_upos)]

                combined_predictions[possible_word] = predicted_words[i][1]*upos_confidence
            return combined_predictions

    def train(self, filename):
        self.word_predictor_trainer.train(filename)
        #self.tag_predictor_trainer.train(filename)

    def trial(self, filename):
        tagger = POSClient(host='localhost', port=9198)
        upos_1h_labels = tag_predictor.get_upos_1h_labels()

        word_vectors = KeyedVectors.load(filename+".wv", mmap='r')

        while True:
            to_predict = input("Text: ")
            combined_predictions = self.predict(to_predict, tagger, upos_1h_labels, word_vectors)
            
            if combined_predictions:
                for predicted_word, predicted_probability in sorted(combined_predictions.items(), key=itemgetter(1), reverse=True):
                    print(f"Predicted '{predicted_word}' with {predicted_probability*100}% confidence")

    def test(self, corpora):
        nlp = stanfordnlp.Pipeline()

        correct = 0
        incorrect = 0

        for i, test_trial in tqdm(enumerate(ngrams[0:len(ngrams)-2])):
            next_word = ngrams[i+1].split(' ')[-1]
            
            combined_predictions = self.predict(test_trial, nlp)
            if combined_predictions:
                predicted_word, predicted_probability  = sorted(combined_predictions.items(), key=itemgetter(1), reverse=True)[0]

                if predicted_word == next_word:
                    correct += 1
                else:
                    incorrect += 1

        return correct, incorrect