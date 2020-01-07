from cleverhans.attacks import SaliencyMapMethod

from tools.cleverhans.adversarial_attack import AdversarialAttack


class SaliencyMapAttack(AdversarialAttack):
    def __init__(self, model, theta=1.0, gamma=1.0, clip_min=-1.0, clip_max=1.0, targeted=False, symbolic_impl=True):
        super().__init__(model=model, clip_min=clip_min, clip_max=clip_max)
        self._targeted = targeted
        self._theta = theta
        self._gamma = gamma
        self._symbolic_impl = symbolic_impl

        with self.graph.as_default():
            self._method = SaliencyMapMethod(self._model, sess=self.session, theta=self._theta, gamma=self._gamma,
                                             nb_classes=self._n_classes, clip_min=self._clip_min,
                                             clip_max=self._clip_max, symbolic_impl=self._symbolic_impl)

    def attack_method(self, labels):
        if self._targeted:
            return self._method.generate(x=self._x_clean, y_target=labels)
        return self._method.generate(x=self._x_clean)
