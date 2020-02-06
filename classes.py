from abc import ABC, abstractmethod
import os
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.callbacks import Callback

from typing import List, Dict, Optional


class Implementation(ABC):
    def __init__(self):
        self.trainer = None

    @abstractmethod
    def test(self, filename: str):
        return NotImplementedError

    @abstractmethod
    def trial(self, filename: str):
        return NotImplementedError

    @abstractmethod
    def train(self, filename: str):
        return NotImplementedError


class Trainer(ABC):
    @abstractmethod
    def train(self, filename: str):
        return NotImplementedError


class TrainerML(Trainer, ABC):
    def __init__(self, n: int, filenames: Dict[str, str], loss: str, optimizer: str):
        """

        @param n: constant of n-grams
        @param filenames: filenames to use
        @param loss: loss function for model to use
        @param optimizer: optimizer for model to use
        """
        super().__init__()
        self.n: int = n

        self.filenames: Dict[str, str] = filenames

        self.loss: str = loss
        self.optimizer: str = optimizer

        self.model: Optional[Sequential] = None

    def load_model(self) -> bool:
        """Loads model structure and weights
        @return: bool if load successful
        """
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
        """Saves model structure and weights
        """
        os.makedirs(os.path.dirname(self.filenames["model_json"]), exist_ok=True)        
        with open(self.filenames["model_json"], 'w') as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(self.filenames["model_weights"])

    @abstractmethod
    def create_model(self):
        return NotImplementedError

    @abstractmethod
    def init_model(self):
        return NotImplementedError


class ModelSaver(Callback):
    def __init__(self, filename: str):
        super().__init__()
        self.filename: str = filename

    def on_train_batch_end(self, batch: int, logs: object = None):
        """

        @param batch: index of batch
        @param logs:
        """
        if logs is None:
            logs = {}

        # Save on every 1000th batch
        if batch % 1000 == 0:
            self.model.save_weights(self.filename.format(batch))
