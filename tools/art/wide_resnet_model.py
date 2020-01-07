from models.original_models import wide_resnet_model
from tools.art.model import Model


class WideResNetModel(Model):
    def __init__(self, checkpoint_path, image_height=32, image_width=32, n_channels=3, n_classes=10):
        super().__init__(image_height=image_height, image_width=image_width, n_channels=n_channels, n_classes=n_classes,
                         checkpoint_path=checkpoint_path)

    def calculate_logits(self, inputs):
        model = wide_resnet_model.WideResnetModel(image_height=self.image_height, image_width=self.image_width,
                                                  n_channels=self.n_channels, n_classes=self.n_classes)
        output = model.fprop(inputs)
        logits = output['logits']
        return logits
