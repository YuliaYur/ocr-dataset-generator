from abc import ABC
from typing import Callable, Optional

import numpy as np


class BaseImageOperation(ABC):
    """
    Base class for all image operations. Provides an interface for successor classes. 
    """
    
    _op: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def process(self, image: np.ndarray) -> np.ndarray:
        if not self._op:
            raise NotImplementedError
        
        return self._op(image)

    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        if not self._op:
            raise NotImplementedError
        
        return self._op(image)
