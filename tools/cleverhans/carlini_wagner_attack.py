from cleverhans.attacks import CarliniWagnerL2

from tools.cleverhans.adversarial_attack import AdversarialAttack


class CarliniWagnerAttack(AdversarialAttack):
    def __init__(self, model, targeted=False, confidence=0, batch_size=1, learning_rate=5e-3, binary_search_steps=5,
                 max_iterations=1000, abort_early=True, initial_const=1e-2, clip_min=-1, clip_max=1):
        super().__init__(model=model, clip_min=clip_min, clip_max=clip_max)
        self._targeted = targeted
        self._confidence = confidence
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._binary_search_steps = binary_search_steps
        self._max_iterations = max_iterations
        self._abort_early = abort_early
        self._initial_const = initial_const

        with self.graph.as_default():
            self._method = CarliniWagnerL2(self._model, sess=self.session, confidence=self._confidence,
                                           batch_size=self._batch_size, learning_rate=self._learning_rate,
                                           binary_search_steps=self._binary_search_steps,
                                           max_iterations=self._max_iterations, abort_early=self._abort_early,
                                           initial_const=self._initial_const, clip_min=self._clip_min,
                                           clip_max=self._clip_max, targeted=self._targeted)

    def get_name(self):
        return "{}_{}".format(self.TOOL_NAME, "C&W")

    def attack_method(self, labels):
        if labels is not None:
            if self._targeted:
                return self._method.generate(x=self._x_clean, y_target=labels)
            else:
                return self._method.generate(x=self._x_clean, y=labels)
        return self._method.generate(x=self._x_clean)
