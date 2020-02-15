from typing import Dict, List
import ujson
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import Sequence

import classes

# Define constant tuple of UPOS tags
UPOS_TAGS = ("CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN",
             "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM",
             "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "$")
upos_1h_labels = None


def get_upos_1h_labels() -> Dict[str, List[int]]:
    """
    Create one-hot encoded upos tags if global variable not set
    @return: dict of one-hot encoded values keyed by upos tags
    """
    global upos_1h_labels
    if upos_1h_labels is not None:
        return upos_1h_labels
    else:
        # Use pandas library to generate upos one-hort encoded values
        upos_1h = pd.get_dummies(UPOS_TAGS).values
        upos_1h_labels: Dict[str, List[int]] = {UPOS_TAGS[i]: v for (i, v) in enumerate(upos_1h)}
        return upos_1h_labels


class TagPredictorDataGenerator(Sequence):
    """
    Provides class to sequentially provide X Y data during training of Tag Predictor Model
    """
    def __init__(self, filename: str, n: int, batch_size: int = 32):
        self.batch_size: int = batch_size
        with open(filename, 'r') as f:
            json = ujson.load(f)
            self.upos_length: int = json["len"]
            self.upos_tags: List[str] = json["upos"]

        self.upos_1h_labels: Dict[str, List[int]] = get_upos_1h_labels()
        self.upos_shape = np.array(list(
            self.upos_1h_labels.values()
        ))[0].shape

        self.index: int = 0
        self.n: int = n

    def __len__(self) -> int:
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
        self.model.add(LSTM(50, input_shape=(self.n, len(UPOS_TAGS))))
        self.model.add(Dense(len(UPOS_TAGS), activation="softmax"))

    def train(self, filename):
        self.write_model()
        print("Training model...")
        training_generator = TagPredictorDataGenerator(filename + ".upos", self.n)

        callbacks_list = [classes.ModelSaver(self.filenames["model_weights"])]

        self.model.fit_generator(generator=training_generator, callbacks=callbacks_list)

        print("Writing model to {}".format(self.filenames["model_json"]))
        self.write_model()
