from models.original_models import wide_resnet_model
from models.ready_models.tf_model import TFModel


class WideResNetModel(TFModel):
    def __init__(self, image_height=32, image_width=32, n_channels=3, n_classes=10, batch_size=32,
                 checkpoint_path="./ready_models/wide_resnet_ckpt"):
        super().__init__(image_height=image_height, image_width=image_width, n_channels=n_channels, n_classes=n_classes,
                         batch_size=batch_size, checkpoint_path=checkpoint_path)

    def calculate_logits(self, inputs):
        model = wide_resnet_model.WideResnetModel(image_height=self.image_height, image_width=self.image_width,
                                                  n_channels=self.n_channels, n_classes=self.n_classes)
        output = model.fprop(inputs)
        logits = output['logits']
        return logits
