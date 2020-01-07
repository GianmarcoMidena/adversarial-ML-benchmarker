import numpy as np
from foolbox.attacks import SaliencyMapAttack as SaliencyMapAttackDef
from foolbox.criteria import Misclassification
from foolbox.distances import MSE

from tools.foolbox.adversarial_attack import AdversarialAttack


class SaliencyMapAttack(AdversarialAttack):
    def __init__(self, model, theta=0.1, min_perturbation=None, max_iterations=2000, num_random_targets=0, fast=True,
                 max_perturbations_per_pixel=7, criterion=Misclassification(), distance=MSE):
        super().__init__(attack_method_def=SaliencyMapAttackDef, model=model, min_perturbation=min_perturbation,
                         criterion=criterion, distance=distance)
        self._max_iterations = max_iterations
        self._num_random_targets = num_random_targets
        self._fast = fast
        self._theta = theta
        self._max_perturbations_per_pixel = max_perturbations_per_pixel

    def apply_attack_method(self, x, y=None):
        batch_size = x.shape[0]
        adv_images = np.zeros(x.shape)
        for i in range(batch_size):
            adv_image = self.attack_method(x[i], label=y[i], unpack=True, max_iter=self._max_iterations,
                                           num_random_targets=self._num_random_targets, fast=self._fast,
                                           theta=self._theta,
                                           max_perturbations_per_pixel=self._max_perturbations_per_pixel)
            if adv_image is None:
                adv_image = x[i]
            adv_images[i] += adv_image
        return adv_images
