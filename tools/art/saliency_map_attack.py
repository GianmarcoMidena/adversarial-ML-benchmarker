from art.attacks import SaliencyMapMethod

from tools.art.adversarial_attack import AdversarialAttack


class SaliencyMapAttack(AdversarialAttack):
    def __init__(self, model, theta=0.1, gamma=1.0, batch_size=16):
        super().__init__(model=model)
        self._theta = theta
        self._gamma = gamma
        self._method = SaliencyMapMethod(classifier=self.model, theta=self._theta, gamma=self._gamma,
                                         batch_size=batch_size)

    def attack_method(self, x, y=None):
        params = {}
        if y is not None:
            params['y'] = y
        return self._method.generate(x=x, **params)
