from tools.foolbox.wide_resnet_model import WideResNetModel
from tools.foolbox.inception_v3_model import InceptionV3Model
from tools.foolbox.fgm_attack import FGMAttack
from tools.foolbox.deep_fool_attack import DeepFoolAttack
from tools.foolbox.carlini_wagner_attack import CarliniWagnerAttack
from tools.foolbox.bim_attack import BIMAttack
from tools.foolbox.madry_model import MadryModel
from tools.foolbox.pgd_attack import PGDAttack
from tools.foolbox.saliency_map_attack import SaliencyMapAttack


ATTACKS_BY_NAME = {
    "fgm": FGMAttack,
    "deepfool": DeepFoolAttack,
    "c&w": CarliniWagnerAttack,
    "bim": BIMAttack,
    "pgd": PGDAttack,
    "saliencymap": SaliencyMapAttack
}


class AdversarialAttacks:
    @classmethod
    def create_attack(cls, attack_name, dataset_name, image_height, image_width, n_channels, n_classes, checkpoint_path):
        attack = cls._find_attack(attack_name)
        model = cls._get_model(dataset_name=dataset_name, image_height=image_height, image_width=image_width,
                               n_channels=n_channels, n_classes=n_classes, checkpoint_path=checkpoint_path)
        return attack(model=model)

    @staticmethod
    def _find_attack(attack_name):
        if attack_name.lower() in ATTACKS_BY_NAME:
            return ATTACKS_BY_NAME[attack_name.lower()]
        raise Exception(f"Attention: \"{attack_name}\" is not an available attack for CleverHans tool!")

    @staticmethod
    def _get_model(dataset_name, image_height, image_width, n_channels, n_classes, checkpoint_path):
        dataset_name = dataset_name.lower()
        if dataset_name == 'imagenet':
            return InceptionV3Model(image_height=image_height, image_width=image_width, n_channels=n_channels,
                                    n_classes=n_classes, checkpoint_path=checkpoint_path)
        elif dataset_name == 'cifar10':
            return WideResNetModel(image_height=image_height, image_width=image_width, n_channels=n_channels,
                                   n_classes=n_classes, checkpoint_path=checkpoint_path)
        elif dataset_name == 'mnist':
            return MadryModel(image_height=image_height, image_width=image_width, n_channels=n_channels,
                              n_classes=n_classes, checkpoint_path=checkpoint_path)
        else:
            raise Exception("{} dataset is not compatible!".format(dataset_name))
