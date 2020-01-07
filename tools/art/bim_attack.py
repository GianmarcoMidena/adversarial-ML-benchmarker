from art.attacks import BasicIterativeMethod

from tools.art.adversarial_attack import AdversarialAttack


class BIMAttack(AdversarialAttack):
    def __init__(self, model, step_size_iter=0.1, max_perturbation=0.3, max_iterations=100, targeted=False,
                 batch_size=16):
        super().__init__(model=model)
        self._targeted = targeted
        self._step_size_iter = step_size_iter
        self._max_perturbation = max_perturbation
        self._max_iterations = max_iterations
        self._method = BasicIterativeMethod(classifier=self.model, targeted=self._targeted, eps=self._max_perturbation,
                                            eps_step=self._step_size_iter, max_iter=self._max_iterations,
                                            batch_size=batch_size)

    def attack_method(self, x, y=None):
        params = {}
        if y is not None:
            params['y'] = y
        return self._method.generate(x=x, **params)
