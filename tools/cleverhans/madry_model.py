from models.original_models import madry_model as model
from tools.cleverhans.model import Model


class MadryModel(Model):
    def __init__(self, checkpoint_path, image_height, image_width, n_channels, n_classes):
        super().__init__(image_height=image_height, image_width=image_width, n_channels=n_channels, n_classes=n_classes,
                         checkpoint_path=checkpoint_path)
        self._model = None

    def fprop(self, x):
        return self._get_model().fprop(x)

    def _get_model(self):
        if self._model is None:
            self._model = model.MadryModel(n_classes=self.n_classes)
        return self._model

    def get_layer_names(self):
        pass

    def make_input_placeholder(self):
        pass

    def make_label_placeholder(self):
        pass
