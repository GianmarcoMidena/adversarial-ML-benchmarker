from models.original_models import wide_resnet_model as model
from tools.cleverhans.model import Model


class WideResNetModel(Model):
    def __init__(self, checkpoint_path, image_height=32, image_width=32, n_channels=3, n_classes=10):
        super().__init__(image_height=image_height, image_width=image_width, n_channels=n_channels, n_classes=n_classes,
                         checkpoint_path=checkpoint_path)
        self._model = None

    def fprop(self, x, set_ref=False):
        return self._get_model().fprop(x, set_ref)

    def make_input_placeholder(self):
        return self._get_model().make_input_placeholder()

    def make_label_placeholder(self):
        return self._get_model().make_label_placeholder()

    def _get_model(self):
        if self._model is None:
            self._model = model.WideResnetModel(image_height=self.image_height, image_width=self.image_width,
                                                n_channels=self.n_channels, n_classes=self.n_classes)
        return self._model

    def get_layer_names(self):
        pass
