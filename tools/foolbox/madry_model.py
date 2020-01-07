from models.original_models import madry_model
from tools.foolbox.model import Model


class MadryModel(Model):
    def __init__(self, checkpoint_path, image_width=28, image_height=28, n_channels=1, n_classes=10):
        super().__init__(image_height=image_height, image_width=image_width, n_channels=n_channels, n_classes=n_classes,
                         checkpoint_path=checkpoint_path)

    def calculate_logits(self, inputs):
        model = madry_model.MadryModel(n_classes=self.n_classes)
        output = model.fprop(inputs)
        return output['logits']
