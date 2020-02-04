from abc import ABC, abstractmethod
import os
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.callbacks import Callback

from typing import List, Dict


class Implementation(ABC):
    def __init__(self):
        self.trainer = None

    @abstractmethod
    def test(self, filename: str):
        return NotImplementedError

    @abstractmethod
    def trial(self):
        return NotImplementedError


class Trainer(ABC):
    def __init__(self, name: str):
        print(f"Instantiating Trainer {name}")
        self.name: str = name

    @abstractmethod
    def train(self, filename: str):
        return NotImplementedError


class TrainerML(Trainer):
    def __init__(self, name: str):
        super().__init__(name)

        self.model: Sequential = None
        self.filenames: Dict[str, str] = None
        self.X = None
        self.Y = None

    def load_model(self):
        loaded: bool = False
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


class ModelSaver(Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}

        if batch % 1000 == 0:
            self.model.save_weights(self.filename.format(batch))
