import numpy as np
from cleverhans.attacks import FastGradientMethod

from tools.cleverhans.adversarial_attack import AdversarialAttack


class FGMAttack(AdversarialAttack):
    def __init__(self, model, targeted=False, step_size=0.3, norm_order=np.inf, clip_min=None, clip_max=None,
                 sanity_checks=True):
        super().__init__(model=model, clip_min=clip_min, clip_max=clip_max)
        self._step_size = step_size
        self._targeted = targeted
        self._norm_order = norm_order
        self._sanity_checks = sanity_checks

        with self.graph.as_default():
            self._method = FastGradientMethod(self._model, sess=self.session, eps=self._step_size, ord=self._norm_order,
                                              clip_min=self._clip_min, clip_max=self._clip_max,
                                              sanity_checks=self._sanity_checks)

    def attack_method(self, labels):
        if labels is not None:
            if self._targeted:
                return self._method.generate(x=self._x_clean, y_target=labels)
            else:
                return self._method.generate(x=self._x_clean, y=labels)
        return self._method.generate(x=self._x_clean)
