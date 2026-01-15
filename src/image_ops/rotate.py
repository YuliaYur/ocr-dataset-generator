from .base import BaseImageOperation
from ..transformations import rotate


class RotateOperation(BaseImageOperation):
    """
    Class that implements operation of rotating an image
    """
    def __init__(self, angle, center: (int, int), border_value: int = 255):
        self._op = lambda X: rotate(X, angle, center, border_value=border_value)
