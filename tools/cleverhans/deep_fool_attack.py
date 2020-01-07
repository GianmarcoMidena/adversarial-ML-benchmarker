from cleverhans.attacks import DeepFool

from tools.cleverhans.adversarial_attack import AdversarialAttack


class DeepFoolAttack(AdversarialAttack):
    def __init__(self, model, n_candidates=10, overshoot=0.02, max_iterations=50, clip_min=-1., clip_max=1.):
        super().__init__(model=model, clip_min=clip_min, clip_max=clip_max)
        self._n_candidate = n_candidates
        self._overshoot = overshoot
        self._max_iterations = max_iterations

        with self.graph.as_default():
            self._method = DeepFool(self._model, sess=self.session, nb_candidate=self._n_candidate,
                                    overshoot=self._overshoot, max_iter=self._max_iterations,
                                    nb_classes=self.n_classes, clip_min=self._clip_min, clip_max=self._clip_max)

    def attack_method(self, labels):
        return self._method.generate(x=self._x_clean)
