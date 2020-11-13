from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
import numpy as np


@task(returns=np.ndarray)
def einsum(*operands, **kwargs):
    return np.einsum(*operands, **kwargs)


@task(returns=np.ndarray)
def tensordot(x, y, axes):
    return np.tensordot(x, y, axes=axes)


@task(returns=np.ndarray)
def transpose(x, axes):
    return np.transpose(x, axes)





