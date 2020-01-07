from foolbox.attacks import GradientAttack
from foolbox.criteria import Misclassification
from foolbox.distances import MSE

from tools.foolbox.adversarial_attack import AdversarialAttack


class FGMAttack(AdversarialAttack):
    def __init__(self, model, min_perturbation=None, max_perturbation=1, max_steps=1000, criterion=Misclassification(),
                 distance=MSE):
        super().__init__(attack_method_def=GradientAttack, model=model, min_perturbation=min_perturbation,
                         criterion=criterion, distance=distance)
        self._max_steps = max_steps
        self._max_perturbation = max_perturbation

    def apply_attack_method(self, x, y=None):
        return self.attack_method(x, labels=y, unpack=True, epsilons=self._max_steps,
                                  max_epsilon=self._max_perturbation)
