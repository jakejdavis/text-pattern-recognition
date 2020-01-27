from abc import ABC, abstractmethod
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback
import ujson
import string

CORPORA_PATH = "/Users/jakedavis/Documents/School/Computer Science/NEA/Predict/Production/corpora/en_US.blogs.txt"

class Implementation(ABC):
    def __init__(self):
        self.trainer = None

    @abstractmethod
    def test(self): 
        pass

    @abstractmethod
    def trial(self):
        pass

class Trainer(ABC):
    def __init__(self, name):
        print(f"Instantiating Trainer {name}")
        self.name = name

    @abstractmethod
    def train(self):
        return NotImplementedError

class TrainerML(Trainer):
    def __init__(self):
        self.filenames = None
        self.X = None
        self.Y = None

    def load_model(self):
        loaded = False
        if os.path.isfile(self.filenames["model_json"]):
            
            print("Loading model from {}".format(self.filenames["model_json"]))
            with open(self.filenames["model_json"], 'r') as model_json:
                self.model = model_from_json(model_json.read()) 
                loaded = True

            # Try to load weights
            if os.path.isfile(self.filenames["model_weights"]):
                print("Loading weights from {}".format(self.filenames["model_weights"]))
                self.model.load_weights(self.filenames["model_weights"])
                
        return loaded
    
    def write_model(self):
        os.makedirs(os.path.dirname(self.filenames["model_json"]), exist_ok=True)        
        with open(self.filenames["model_json"], 'w') as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(self.filenames["model_weights"])

    def train(self, filename, batch_size, epochs):
        
        print("Training model...")
        self.model.fit(self.X, self.Y, batch_size=batch_size, epochs=epochs)

        print("Writing model to {}".format(self.filenames["model_json"]))
        self.write_model()


class ModelSaver(Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_train_batch_end(self, batch, logs={}):
        if batch % 1000 == 0: 
            self.model.save_weights(self.filename.format(batch))