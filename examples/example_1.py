from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from vznncv.signal.generator.onedim import create_process_realization


def f_psd_gaussian(f, f_0, alpha):
    """
    One side gaussian psd function

    .. math::

       s = \sqrt{\frac{1}{4 \pi \alpha}} \left(
       e^{-\frac{\left( f - f_0 \right) ^ 2}{4 \alpha}}
       \right)

    :param f:
    :param f_0:
    :param alpha:
    :return:
    """
    return np.sqrt(1 / (4 * np.pi * alpha)) * (
        np.exp(-(f - f_0) ** 2 / (4 * alpha))
    )


fs = 2.0
t = np.arange(200) / fs

x = create_process_realization(
    size=t.size,
    f_psd=partial(f_psd_gaussian, f_0=0.002, alpha=0.001),
    f_m=0.0,
    f_std=2.0,
    fs=2.0,
)

plt.plot(t, x)
plt.xlabel('x')
plt.ylabel('t')
plt.show()
