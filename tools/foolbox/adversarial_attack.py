from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class AdversarialAttack(ABC):
    TOOL_NAME = "Foolbox"

    def __init__(self, attack_method_def, model, min_perturbation, criterion, distance):
        self._attack_method_def = attack_method_def
        self._model = model
        self._min_perturbation = min_perturbation
        self._criterion = criterion
        self._distance = distance
        self._image_height = model.image_height
        self._image_width = model.image_width
        self._n_channels = model.n_channels
        self._n_classes = model.n_classes
        self._attack_method = None

        graph = tf.Graph()
        session = tf.compat.v1.Session(graph=graph)
        with graph.as_default():
            with session.as_default() as sess:
                self._model = self._model(sess)
                self._attack_method = self._attack_method_def(model=self._model, criterion=self._criterion,
                                                              distance=self._distance, threshold=self._min_perturbation)

    @property
    def attack_method(self):
        return self._attack_method

    def get_name(self):
        return "{}_{}".format(self.TOOL_NAME, self.__class__.__name__.replace("Attack", ""))

    def execute(self, x, y=None):
        if y is None:
            y = np.argmax(self._model.forward(x), axis=-1)
        x_adv = self.apply_attack_method(x, y)
        return x_adv

    @abstractmethod
    def apply_attack_method(self, x, y=None):
        ...
