import numpy as np
from foolbox.attacks import DeepFoolL2Attack
from foolbox.criteria import Misclassification
from foolbox.distances import MSE

from tools.foolbox.adversarial_attack import AdversarialAttack


class DeepFoolAttack(AdversarialAttack):
    def __init__(self, model, min_perturbation=None, max_iterations=100, subsample=10, criterion=Misclassification(),
                 distance=MSE):
        super().__init__(attack_method_def=DeepFoolL2Attack, model=model, min_perturbation=min_perturbation,
                         criterion=criterion, distance=distance)
        self._max_iterations = max_iterations
        self._subsample = subsample

    def apply_attack_method(self, x, y=None):
        batch_size = x.shape[0]
        adv_images = np.zeros(x.shape)
        for i in range(batch_size):
            adv_images[i] += self.attack_method(x[i], label=y[i], unpack=True, steps=self._max_iterations,
                                                subsample=self._subsample)
        return adv_images
