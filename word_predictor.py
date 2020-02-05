from gensim.models import KeyedVectors
import ujson
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import Sequence
import numpy as np

import classes


class WordPredictorDataGenerator(Sequence):
    def __init__(self, words_filename, kv_filename, n, batch_size=32):
        self.batch_size = batch_size
        with open(words_filename, 'r') as f:
            json = ujson.load(f)
            self.texts_length = json["len"]
            self.texts = json["texts"]
        self.word_vectors = KeyedVectors.load(kv_filename, mmap='r')

        self.index = 0
        self.n = n

    def __len__(self):
        return int(np.floor(self.texts_length / 3 / self.batch_size))

    def __getitem__(self, index):
        # Generates batch
        X = np.zeros([self.batch_size, self.n, self.word_vectors.vector_size])
        Y = np.zeros([self.batch_size, self.word_vectors.vector_size])

        i = 0
        # Generate X Y data until full (last item is not zero'd)
        while np.count_nonzero(X[self.batch_size - 1]) == 0:
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
            self.word_vectors[v] for v in self.texts[self.index:self.index + self.n]
        ])
        y = np.array([self.word_vectors[self.texts[self.index + self.n]]])

        return x, y


class WordPredictorTrainer(classes.TrainerML):
    def __init__(self, n, filenames, loss, optimizer):
        super().__init__(n, filenames, loss, optimizer)

    def create_model(self):
        word_vectors = KeyedVectors.load(self.filenames["word_vectors"], mmap='r')
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(3, word_vectors.vector_size), return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(word_vectors.vector_size, activation='tanh'))

    def init_model(self):
        if not self.load_model():
            print("Loading model failed... creating model instead")
            self.create_model()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        print("Model compiled!")

    def train(self, filename):
        print("Training model...")
        training_generator = WordPredictorDataGenerator(filename + ".uts", filename + ".wv", self.n)

        callbacks_list = [classes.ModelSaver(self.filenames["model_weights"])]

        self.model.fit_generator(generator=training_generator, callbacks=callbacks_list)

        print("Writing model to {}".format(self.filenames["model_json"]))
        self.write_model()
        self.write_model()