class curriculum_example:

    def __init__(self, img_id, difficulty):
        self._img_id = img_id
        # self._label = label
        # self._pred_confidence = pred_confidence
        # self._pred_p = pred_p
        # self._grad_magnitude = grad_magnitude
        # self._grad_p_magnitude = grad_p_magnitude
        self._difficulty = difficulty

    @property
    def img_id(self):
        return self._img_id

    # @property
    # def pred_confidence(self):
    #     return self._pred_confidence
    #
    # @property
    # def pred_p(self):
    #     return self._pred_p
    #
    # @property
    # def grad_magnitude(self):
    #     return self._grad_magnitude
    #
    # @property
    # def grad_p_magnitude(self):
    #     return self._grad_p_magnitude

    @property
    def difficulty(self):
        return self._difficulty
