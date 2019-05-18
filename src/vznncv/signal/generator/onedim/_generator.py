"""
Main module to generate one dimensional random process.
"""
import logging
from typing import Iterable

import numpy as np

from vznncv.signal.generator.onedim._window import calculate_linear_window_size, linear_window
from ._utils import get_random_state, UnaryFunction, BinaryFunction, TrendFunction

_DEFAULT_DTYPE = np.float64

logger = logging.getLogger(__name__)


class _BaseFrameGenerator:
    """
    Fixed length random process generator, that is based on the FFT.
    """
    _DOUBLE_PI = 2 * np.pi

    def __init__(self, frame_size, fs=1.0, dtype=_DEFAULT_DTYPE, random_state=None):
        self.freq = np.fft.rfftfreq(frame_size, 1 / fs).astype(dtype)
        self._random = get_random_state(random_state)
        self._fs = fs

        self._harmonics = np.empty_like(self.freq, dtype=np.result_type(dtype, np.complex64))
        self._a = np.empty_like(self.freq, dtype=dtype)

    def generate(self, s):
        if s.shape != self.freq.shape:
            raise ValueError("Invalid s shape {}. Expected {}".format(s.shape, self.freq.shape))

        self._harmonics.real = self._random.randn(self.freq.size)
        self._harmonics.imag = self._random.randn(self.freq.size)
        # get harmonic amplitudes from one side psd
        s = np.sqrt(s)
        np.divide(s, np.sqrt(2), out=self._a)
        self._a *= np.sqrt(self._fs / 2)

        self._harmonics *= self._a

        frame = np.fft.irfft(self._harmonics, norm='ortho')
        return frame.real


#: minimal window size, when "auto" option is used
_MINIMAL_WINDOW_SIZE = 8192


def _build_f_psd(f_psd, fs):
    if f_psd is None:
        new_f_psd = lambda f, t: np.full_like(f, 1)
        new_f_psd.stationary = True
    elif isinstance(f_psd, UnaryFunction):
        new_f_psd = lambda f, t: f_psd(np.abs(f))
        new_f_psd.stationary = True
    elif isinstance(f_psd, BinaryFunction):
        new_f_psd = lambda f, t: f_psd(np.abs(f), t)
        new_f_psd.stationary = False
    else:
        raise ValueError("f_psd isn't None, unary or binary function: {}".format(f_psd))
    return new_f_psd


def _iterate_time_frames(frame_size, t_0=0.0, fs=1.0, dtype=_DEFAULT_DTYPE):
    dt = 1.0 / fs
    frame = np.arange(frame_size, dtype=dtype) * dt + t_0
    d_frame = np.full_like(frame, fill_value=dt * len(frame))
    while True:
        yield frame
        frame += d_frame


def generate_process_realization(*, f_psd=None, f_m=None, f_std=None, fs=1.0,
                                 dtype=_DEFAULT_DTYPE, random=None,
                                 precision=0.99, window_size='auto') -> Iterable[np.array]:
    """
    Generate random process with a given power spectral density, mean, and standard derivation.

    The function yields successive array of the same realizations. A length of the result blocks is undefined,
    but it's guarantee that all blocks has the same length.

    Parameters
    ----------

    f_psd
         power spectral density function that can be the following:
         - ``None`` - the white noise psd will be used;
         - ``f_psd(f)`` - the function with one positional argument. In this case ``f_psd`` isn't changed
           over time;
         - ``f_psd(f, t) - the function with two positional arguments. In this case ``f_psd`` can be changed
           over time.
         For last 2 cases ``f_psd`` should be defined for ``0 <= f <= fs/2``. The ``f`` argument is numpy
         array, ``t`` - scalar.
    f_m
        mean value function:
        - ``None`` - mean will be zero.
        - scalar - mean is specified constant and isn't changed over time.
        - ``f_m(t)`` - mean value is changed over time according this function. The function should accept
                       numpy array
        - ``f_m(t, out)`` - like previous option, but it has an ``out`` parameter for an output array
                            for a memory optimization
    f_std
        standard derivation function:
        - ``None`` - standard derivation will be set by the ``f_psd``.
        - scalar - the ``f_psd`` will be normalized, and standard derivation will have a defined value.
        - ``f_std(t)`` -  the ``f_psd`` will be normalized, and standard derivation will be changed according
                          ``f_std`` function.
        - ``f_std(t, out)`` - like previous option, but has an ``out`` parameter for an output array
                              for a memory optimization
    fs
        sampling frequency
    dtype
        result type
    random
        :class:`numpy.random.RandomState` object or its seed
    precision
        expected precision of the random characteristics

    Returns
    -------
    Iterable[np.array]
        generator of the random process realization blocks


    Notes
    -----

    The generator efficiently creates a realization of any length using gluing of independent random process
    implementations with 50% overlapping. It reduces a memory consumption, but adds distortion into a
    power spectral density (PSD) of a result process.

    The acceptable PSD precision for a stationary spectrum is specified with ``precision`` parameter.
    The precision can be set only if ``window_size`` is set to ``'auto'``.

    If ``window_size`` is set to positive number or PSD is changed over time, the specified ``precision`` isn't
    guaranteed.
    """
    # check f_m
    f_m = TrendFunction(f_m, default_value=0.0)

    # check f_std
    normalize_psd = f_std is not None
    f_std = TrendFunction(f_std, default_value=1.0)

    # check f_psd
    f_psd = _build_f_psd(f_psd, fs)

    # calculate minimal window size
    if window_size == 'auto':
        window_size = calculate_linear_window_size(
            f_psd=lambda f: f_psd(f, 0),
            precision=precision,
            fs=fs
        )

        # increase window size for efficiency
        if window_size < _MINIMAL_WINDOW_SIZE:
            window_size = _MINIMAL_WINDOW_SIZE
    # make window size even to simplify frame overlap
    if window_size % 2 == 1:
        window_size += 1

    logger.debug("Set window_size to {}".format(window_size))

    # create window to glue frames
    window = linear_window(window_size)

    # initialize base frame generator
    base_frame_generator = _BaseFrameGenerator(frame_size=len(window), fs=fs, dtype=dtype, random_state=random)
    freq = base_frame_generator.freq

    # normalize f_std if it's needed
    if normalize_psd:
        original_f_psd = f_psd

        def normalized_f_psd(f, t):
            s = original_f_psd(f, t)
            # note: ``f`` is ``freq``
            s_pow = np.trapz(s, freq)
            return s / s_pow

        normalized_f_psd.stationary = f_psd.stationary
        f_psd = normalized_f_psd

    # optimization for a stationary psd
    if f_psd.stationary:
        s = f_psd(freq, 0)
        f_psd = lambda f, t: s
        f_psd.stationary = True

    # generate realizations
    s = f_psd(freq, 0)
    prev_weighted_frame = base_frame_generator.generate(s)
    prev_weighted_frame *= window
    curr_weighted_frame = None
    frame_size = window_size // 2
    m_frame = np.empty(frame_size, dtype=dtype)
    std_frame = np.empty(frame_size, dtype=dtype)

    for time_frame in _iterate_time_frames(frame_size, fs=fs, dtype=dtype):
        s = f_psd(freq, time_frame[-1])
        curr_weighted_frame = base_frame_generator.generate(s)
        curr_weighted_frame *= window

        output_frame = prev_weighted_frame[frame_size:]
        output_frame += curr_weighted_frame[:frame_size]

        f_std(time_frame, out=std_frame)
        f_m(time_frame, out=m_frame)
        output_frame *= f_std(time_frame)
        output_frame += m_frame

        yield output_frame

        prev_weighted_frame = curr_weighted_frame


def create_process_realization(*, size: int, **kwargs):
    """
    The version of the :func:`generate_process_realization` that returns a realization of the specified size ``size``.

    See :func:`generate_process_realization` for more details.
    """
    frame_iter = generate_process_realization(**kwargs)
    frames = []
    total_size = 0
    while total_size < size:
        frame = next(frame_iter)
        total_size += len(frame)
        frames.append(frame)

    return np.hstack(tuple(frames))[:size]
