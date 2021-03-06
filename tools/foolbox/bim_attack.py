from foolbox.attacks import LinfinityBasicIterativeAttack
from foolbox.criteria import Misclassification
from foolbox.distances import MSE

from tools.foolbox.adversarial_attack import AdversarialAttack


class BIMAttack(AdversarialAttack):
    def __init__(self, model, step_size_iter=0.05, max_perturbation=0.3, n_iterations=10, min_perturbation=None,
                 binary_search=True, random_start=False, return_early=True, criterion=Misclassification(),
                 distance=MSE):
        super().__init__(attack_method_def=LinfinityBasicIterativeAttack, model=model,
                         min_perturbation=min_perturbation, criterion=criterion, distance=distance)
        self._binary_search = binary_search
        self._step_size_iter = step_size_iter
        self._n_iterations = n_iterations
        self._random_start = random_start
        self._return_early = return_early
        self._max_perturbation = max_perturbation

    def apply_attack_method(self, x, y=None):
        return self.attack_method(x, labels=y, unpack=True, binary_search=self._binary_search,
                                  epsilon=self._max_perturbation, stepsize=self._step_size_iter,
                                  iterations=self._n_iterations, random_start=self._random_start,
                                  return_early=self._return_early)
