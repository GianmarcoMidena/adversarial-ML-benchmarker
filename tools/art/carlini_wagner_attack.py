from art.attacks import CarliniL2Method

from tools.art.adversarial_attack import AdversarialAttack


class CarliniWagnerAttack(AdversarialAttack):
    def __init__(self, model, targeted=False, confidence=0.0, learning_rate=0.01, binary_search_steps=10,
                 max_iterations=10, initial_const=0.01, max_halving=5, max_doubling=5, batch_size=16):
        super().__init__(model=model)
        self._targeted = targeted
        self._confidence = confidence
        self._learning_rate = learning_rate
        self._binary_search_steps = binary_search_steps
        self._max_iterations = max_iterations
        self._initial_const = initial_const
        self._max_halving = max_halving
        self._max_doubling = max_doubling
        self._method = CarliniL2Method(classifier=self.model, targeted=self._targeted, confidence=self._confidence,
                                       learning_rate=self._learning_rate, binary_search_steps=self._binary_search_steps,
                                       max_iter=self._max_iterations, initial_const=self._initial_const,
                                       max_halving=self._max_halving, max_doubling=self._max_doubling,
                                       batch_size=batch_size)

    def get_name(self):
        return "{}_{}".format(self.TOOL_NAME, "C&W")

    def attack_method(self, x, y=None):
        params = {}
        if y is not None:
            params['y'] = y
        return self._method.generate(x=x, **params)
