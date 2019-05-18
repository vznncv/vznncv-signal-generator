"""
Helper module to build windows for frame gluing.
"""
import logging
from math import ceil

import numpy as np
from scipy import interpolate

_DEFAULT_DTYPE = np.float32

logger = logging.getLogger(__name__)


def calculate_correlation_interval(f_r, alpha=0.01, fs=1.0):
    """
    Calculate correlation interval ``t_k`` that satisfies the following condition:

    ``f_r(t)`` < ``f_r(0) * alpha`` for any ``t`` >= ``t_k``.

    We suppose that ``f_r`` converge to zero.

    :param f_r: autocorrelation function
    :param alpha:
    :param fs: sampling frequency
    :return:
    """
    block_len = 16 * 1024
    dt = 1 / fs

    for _ in range(20):
        t = np.arange(block_len, dtype=_DEFAULT_DTYPE) * dt
        r = f_r(t)
        if np.isnan(r).any():
            raise ValueError("Autocorrelation function returned NaN")
        np.abs(r, out=r)
        np.maximum.accumulate(r[::-1], out=r[::-1])
        i = block_len - np.searchsorted(r[::-1], alpha * r[0])
        if i <= block_len / 5:
            dt /= 2
        elif i <= block_len / 2:
            t_k = dt * i
            break
        else:
            dt *= 2
    else:
        raise ValueError("Cannot find correlation interval")

    return t_k


def build_f_fun_from_f_psd(f_psd, fs=1.0):
    """
    Build autocorrelation function from power spectral density function.

    We assume that ``f_psd(t) = 0`` for any ``t >= fs / 2``

    :param f_psd:
    :param fs:
    :return:
    """
    base_point_n = 4 * 1024
    smooth_k = 8
    smooth_psd_alpha = 500

    s = np.zeros(base_point_n * smooth_k, dtype=_DEFAULT_DTYPE)
    f = np.fft.rfftfreq(base_point_n)
    # suppress Gibbs phenomenon
    smooth_psd_k = 1 - np.exp(-np.abs(np.abs(f[-1] - f)) * smooth_psd_alpha)
    f *= fs
    s[:len(f)] = f_psd(f) * smooth_psd_k
    fs = fs * smooth_k

    if np.isnan(s).any():
        raise ValueError("PSD function returned NaN")
    r = np.fft.irfft(s)
    r = r[:(r.size + 1) // 2]
    r *= 2 * fs

    t = np.arange(len(r)) * (0.5 / fs)
    f_r_os = interpolate.interp1d(t, r, bounds_error=False, fill_value=0)

    def f_r(t):
        t = np.abs(t)
        return f_r_os(t)

    f_r.__doc__ = """
        Correlation function of the {}
        :param t: 
        :return: 
        """.format(f_psd)

    return f_r


def calculate_correlation_interval_for_psd(f_psd, alpha=0.01, fs=1.0):
    """
    Calculate correlation interval ``t_k`` that satisfies the following condition:

    ``f_r(t)`` < ``f_r(0) * alpha`` for any ``t`` >= ``t_k``.

    where ``f_r`` is correlation function that is derived from ``f_psd``.

    We suppose that ``f_r`` converge to zero.

    :param f_psd:
    :param alpha:
    :param fs:
    :return:
    """
    f_r = build_f_fun_from_f_psd(f_psd, fs=fs)
    return calculate_correlation_interval(f_r, alpha=alpha, fs=fs)


def build_common_window_function(monotonic_fun, doc=None):
    """
    Build window function to glue random process realization with a 50% overlap.

    The ``monotonic_fun`` should be define in the range [0, 1].

    :param monotonic_fun: some monotonic function
    :param doc: function documentation
    :return: window function
    """
    # normalize monotonic function
    y_0 = monotonic_fun(0.0)
    y_1 = monotonic_fun(1.0)
    norm_k = 1 / (np.sqrt(2) * (y_1 - y_0))
    norm_monotonic_fun = lambda x: (monotonic_fun(x) - y_0) * norm_k
    # check that function is monotonic
    x = np.linspace(0, 1, num=100)
    y = norm_monotonic_fun(x)
    if not np.all(y[:-1] < y[1:]):
        raise ValueError("The function '{}' is not monotonic".format(monotonic_fun))

    win_power_mirroring = lambda val: np.sqrt(1 - val ** 2)

    def win_fun(win_size):
        """
        Window function.

        :param win_size: number of points in the output window
        :return: array with window coefficients
        """
        half_win_size = (win_size + 1) // 2

        quarter_win_size = (half_win_size + 1) // 2
        if half_win_size % 2 == 0:
            x = np.linspace(0, 1 - 1 / (half_win_size), quarter_win_size)
            quarter_win = norm_monotonic_fun(x)
            half_win = np.hstack((quarter_win, win_power_mirroring(quarter_win[::-1])))
        else:
            x = np.linspace(0, 1, quarter_win_size)
            quarter_win = norm_monotonic_fun(x)
            half_win = np.hstack((quarter_win, win_power_mirroring(quarter_win[-2::-1])))

        # check that source function is monotonic
        if not np.all(quarter_win[1:] > quarter_win[:-1]):
            raise ValueError("Function {} isn't monotonic".format(monotonic_fun))

        if win_size % 2 == 0:
            win = np.hstack((half_win, half_win[::-1]))
        else:
            win = np.hstack((half_win, half_win[-2::-1]))

        return win

    win_fun.__name__ = monotonic_fun.__name__
    if doc:
        win_fun.__doc__ = doc

    return win_fun


linear_window = build_common_window_function(
    lambda x: x, doc=
    """
    Window function to glue random process realization with 50% overlap and minimal PSD distortion.
    """
)


def calculate_linear_window_size(f_psd, precision=0.99, fs=1.0):
    """
    Estimate :fun:`linear_window` size from psd function and given precision.

    :param f_psd:
    :param precision:
    :param fs:
    :return:
    """
    alpha = 1 - precision
    t_k = calculate_correlation_interval_for_psd(f_psd=f_psd, alpha=alpha, fs=fs)
    window_size = (1 / fs) * 2 * t_k / np.sqrt(alpha - alpha ** 2 / 2)
    window_size = int(ceil(window_size))
    return window_size
