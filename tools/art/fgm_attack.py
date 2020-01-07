import numpy as np
from art.attacks import FastGradientMethod

from tools.art.adversarial_attack import AdversarialAttack


class FGMAttack(AdversarialAttack):
    def __init__(self, model, targeted=False, step_size_iter=0.3, max_perturbation=0.1, norm_order=np.inf,
                 num_random_init=0, minimal=False, batch_size=16):
        super().__init__(model=model)
        self._targeted = targeted
        self._step_size_iter = step_size_iter
        self._max_perturbation = max_perturbation
        self._norm_order = norm_order
        self._num_random_init = num_random_init
        self._minimal = minimal

        self._method = FastGradientMethod(classifier=self.model, norm=self._norm_order, eps=self._max_perturbation,
                                          eps_step=self._step_size_iter, targeted=self._targeted,
                                          num_random_init=self._num_random_init, batch_size=batch_size,
                                          minimal=self._minimal)

    def attack_method(self, x, y=None):
        params = {
            'minimal': self._minimal
        }
        if y is not None:
            params['y'] = y
        return self._method.generate(x=x, **params)
