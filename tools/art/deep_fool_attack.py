from art.attacks import DeepFool

from tools.art.adversarial_attack import AdversarialAttack


class DeepFoolAttack(AdversarialAttack):
    def __init__(self, model, overshoot=1e-6, max_iterations=100, n_candidates=10, batch_size=16):
        super().__init__(model=model)
        self._overshoot = overshoot
        self._max_iterations = max_iterations
        self._n_candidates = n_candidates
        self._method = DeepFool(classifier=self.model, epsilon=self._overshoot, max_iter=self._max_iterations,
                                nb_grads=self._n_candidates, batch_size=batch_size)

    def attack_method(self, x, y=None):
        return self._method.generate(x=x)
