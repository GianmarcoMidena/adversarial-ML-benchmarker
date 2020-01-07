from utils.image_io import ImageIO


class NormalizedImageIO(ImageIO):
    @classmethod
    def transform_image(cls, image):
        return ((image / 255.) * 2.) - 1.

    @classmethod
    def inverse_transform_image(cls, image):
        return ((image + 1.) * .5) * 255.
