import numpy as np
from cleverhans.attacks import ProjectedGradientDescent

from tools.cleverhans.adversarial_attack import AdversarialAttack


class PGDAttack(AdversarialAttack):
    def __init__(self, model, targeted=False, step_size_iter=0.05, max_perturbation=0.3, n_iterations=10,
                 norm_order=np.inf, rand_init=None, rand_minmax=0.3, clip_min=None, clip_max=None, sanity_checks=True):
        super().__init__(model=model, clip_min=clip_min, clip_max=clip_max)
        self._targeted = targeted
        self._step_size_iter = step_size_iter
        self._max_perturbation = max_perturbation
        self._n_iterations = n_iterations
        self._norm_order = norm_order
        self._rand_init = rand_init
        self._rand_minmax = rand_minmax
        self._sanity_checks = sanity_checks

        with self.graph.as_default():
            self._method = ProjectedGradientDescent(self._model, sess=self.session, eps=self._max_perturbation,
                                                    eps_iter=self._step_size_iter, nb_iter=self._n_iterations,
                                                    ord=self._norm_order, rand_init=self._rand_init,
                                                    clip_min=self._clip_min, clip_max=self._clip_max,
                                                    sanity_checks=self._sanity_checks)

    def attack_method(self, labels):
        if labels is not None:
            if self._targeted:
                return self._method.generate(x=self._x_clean, y_target=labels, rand_minmax=self._rand_minmax)
            else:
                return self._method.generate(x=self._x_clean, y=labels, rand_minmax=self._rand_minmax)
        return self._method.generate(x=self._x_clean, rand_minmax=self._rand_minmax)
