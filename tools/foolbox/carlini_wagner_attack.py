from foolbox.attacks import CarliniWagnerL2Attack
from foolbox.criteria import Misclassification
from foolbox.distances import MSE

from tools.foolbox.adversarial_attack import AdversarialAttack


class CarliniWagnerAttack(AdversarialAttack):
    def __init__(self, model, min_perturbation=None, binary_search_steps=5, max_iterations=1000, confidence=0,
                 learning_rate=5e-3, initial_const=1e-2, abort_early=True, criterion=Misclassification(),
                 distance=MSE):
        super().__init__(attack_method_def=CarliniWagnerL2Attack, model=model, min_perturbation=min_perturbation,
                         criterion=criterion, distance=distance)
        self._binary_search_steps = binary_search_steps
        self._max_iterations = max_iterations
        self._confidence = confidence
        self._learning_rate = learning_rate
        self._initial_const = initial_const
        self._abort_early = abort_early

    def get_name(self):
        return "{}_{}".format(self.TOOL_NAME, "C&W")

    def apply_attack_method(self, x, y=None):
        return self.attack_method(x, labels=y, unpack=True, binary_search_steps=self._binary_search_steps,
                                  max_iterations=self._max_iterations, confidence=self._confidence,
                                  learning_rate=self._learning_rate, initial_const=self._initial_const,
                                  abort_early=self._abort_early)
