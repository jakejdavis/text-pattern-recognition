import classes
import ujson
import string
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input, Dense, LSTM, Activation
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import pickle
import itertools

upos_1h_labels = None
UPOS_TAGS = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN",
             "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM",
             "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "$"]


def get_upos_1h_labels():
    global upos_1h_labels
    if not upos_1h_labels is None:
        return upos_1h_labels
    else:
        upos_1h = pd.get_dummies(UPOS_TAGS).values
        upos_1h_labels = {UPOS_TAGS[i]: v for (i, v) in enumerate(upos_1h)}
        return upos_1h_labels


class TagPredictorDataGenerator(Sequence):
    def __init__(self, filename, n, batch_size=32):
        self.batch_size = batch_size
        with open(filename, 'r') as f:
            json = ujson.load(f)
            self.upos_length = json["len"]
            self.upos_tags = json["upos"]
        self.upos_1h_labels = get_upos_1h_labels()
        self.upos_shape = np.array(list(self.upos_1h_labels.values()))[0].shape

        self.index = 0
        self.n = n

    def __len__(self):
        return int(np.floor(self.upos_length / 3 / self.batch_size))

    def __getitem__(self, index):
        # Generates batch
        X = np.zeros([self.batch_size, self.n, len(UPOS_TAGS)])
        Y = np.zeros([self.batch_size, len(UPOS_TAGS)])

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
            finally:
                self.index += 1
        return X, Y

    def __data_generation(self):
        x = np.array([
            self.upos_1h_labels[v] for v in self.upos_tags[self.index:self.index + self.n]
        ])
        y = np.array([self.upos_1h_labels[self.upos_tags[self.index + self.n]]])

        return x, y


class TagPredictorTrainer(classes.TrainerML):
    def __init__(self, n, filenames, loss, optimizer):
        super().__init__(n, filenames, loss, optimizer)

    def create_model(self):
        self.model = Sequential()
        print(len(UPOS_TAGS))
        self.model.add(LSTM(50, input_shape=(self.n, len(UPOS_TAGS)), return_sequences=True))
        self.model.add(LSTM(100, return_sequences=False))
        self.model.add(Dense(len(UPOS_TAGS), activation='softmax'))

    def init_model(self):
        if not self.load_model():
            print("Loading model failed... creating model instead")
            self.create_model()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        print("Model compiled!")

    def train(self, filename):
        self.write_model()
        print("Training model...")
        training_generator = TagPredictorDataGenerator(filename + ".upos", self.n)

        callbacks_list = [classes.ModelSaver(self.filenames["model_weights"])]

        self.model.fit_generator(generator=training_generator, callbacks=callbacks_list)

        print("Writing model to {}".format(self.filenames["model_json"]))
        self.write_model()
