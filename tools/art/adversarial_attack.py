import time
from abc import ABC, abstractmethod
import tensorflow as tf


class AdversarialAttack(ABC):
    TOOL_NAME = "ART"

    def __init__(self, model):
        with tf.Graph().as_default() as graph:
            with tf.compat.v1.Session(graph=graph).as_default() as session:
                self._model = model(session)

    @property
    def model(self):
        return self._model

    def get_name(self):
        return "{}_{}".format(self.TOOL_NAME, self.__class__.__name__.replace("Attack", ""))

    def execute(self, x, y=None):
        x_adv = self.attack_method(x, y)
        return x_adv

    @abstractmethod
    def attack_method(self, x, y):
        ...
