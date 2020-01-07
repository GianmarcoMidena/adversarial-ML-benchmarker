from abc import ABC, abstractmethod
import tensorflow as tf


class AdversarialAttack(ABC):
    TOOL_NAME = "CleverHans"

    def __init__(self, model, clip_min, clip_max):
        self._model = model
        self._clip_min = clip_min
        self._clip_max = clip_max
        self._image_height = model.image_height
        self._image_width = model.image_width
        self._n_channels = model.n_channels
        self._n_classes = model.n_classes
        self.graph = tf.compat.v1.Graph()
        self.session = tf.compat.v1.Session(graph=self.graph)

    @property
    def n_classes(self):
        self._n_classes

    def get_name(self):
        return "{}_{}".format(self.TOOL_NAME, self.__class__.__name__.replace("Attack", ""))

    def execute(self, x, y=None):
        with self.graph.as_default():
            self._design_attack(y)

            # var_name_list = [v.name for v in tf.trainable_variables()]
            # print("variable names in code")
            # print(var_name_list)
            #
            # print("keys in checkpoint file")
            # from tensorflow.python import pywrap_tensorflow
            # reader = pywrap_tensorflow.NewCheckpointReader(self._model.checkpoint_path)
            # var_to_shape_map = reader.get_variable_to_shape_map()
            # print(var_to_shape_map)

            saver = tf.compat.v1.train.Saver()
            saver.restore(self.session, self._model.checkpoint_path)
            x_adv = self.session.run(self._x_adv, feed_dict={self._x_clean: x})
        return x_adv

    @abstractmethod
    def attack_method(self, labels):
        ...

    def _design_attack(self, y):
        self._x_clean = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self._image_height, self._image_width, self._n_channels])
        if y is not None:
            labels = tf.compat.v1.placeholder(tf.float32, shape=(None, self._n_classes))
        else:
            labels = None

        self._x_adv = self.attack_method(labels)
