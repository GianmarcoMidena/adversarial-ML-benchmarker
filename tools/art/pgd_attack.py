import numpy as np
from art.attacks import ProjectedGradientDescent

from tools.art.adversarial_attack import AdversarialAttack


class PGDAttack(AdversarialAttack):
    def __init__(self, model, targeted=False, step_size_iter=.1, max_perturbation=.3, norm_order=np.inf,
                 max_iterations=100, num_random_init=0, batch_size=16):
        super().__init__(model=model)
        self._targeted = targeted
        self._step_size_iter = step_size_iter
        self._max_perturbation = max_perturbation
        self._norm_order = norm_order
        self._max_iterations = max_iterations
        self._num_random_init = num_random_init
        self._method = ProjectedGradientDescent(classifier=self.model, targeted=self._targeted, norm=self._norm_order,
                                                eps=self._max_perturbation, eps_step=self._step_size_iter,
                                                max_iter=self._max_iterations, num_random_init=self._num_random_init,
                                                batch_size=batch_size)

    def attack_method(self, x, y=None):
        params = {}
        if y is not None:
            params['y'] = y
        return self._method.generate(x=x, **params)
