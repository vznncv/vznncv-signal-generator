from collections import namedtuple
from unittest import TestCase

import multiprocess as mp
import numpy as np
from scipy import signal
from scipy.special import erf

from testing_utils import assert_almostclose
from vznncv.signal.generator.onedim import generate_process_realization


def f_psd_norm(f, f_0, alpha):
    """
    One side gaussian psd function

    .. math::

       s = \sqrt{\frac{1}{4 \pi \alpha}} \left(
       e^{-\frac{\left( f - f_0 \right) ^ 2}{4 \alpha}} +
       e^{-\frac{\left( f + f_0 \right) ^ 2}{4 \alpha}}
       \right)

    :param f:
    :param f_0:
    :param alpha:
    :return:
    """
    return np.sqrt(1 / (4 * np.pi * alpha)) * (
            np.exp(-(f - f_0) ** 2 / (4 * alpha)) +
            np.exp(-(f + f_0) ** 2 / (4 * alpha))
    )


def build_norm_fun(f_0, alpha):
    """
    Create one side gaussian psd:

    :param f_0:
    :param alpha:
    :return:
    """

    def f_psd(f):
        """
        PSD function of the gaussian function with :math:`\alpha = {alpha:.2f}`

        :param f:
        :return:
        """
        return f_psd_norm(f, f_0=f_0, alpha=alpha)

    f_psd.__doc__ = f_psd.__doc__.format(alpha=alpha)

    return f_psd


def build_sequence(frame_iter, size):
    """
    Build sequence of the specified length from frame iterator.

    :param frame_iter:
    :param size:
    :return:
    """
    frames = []
    total_size = 0
    while total_size < size:
        frame = next(frame_iter)
        total_size += len(frame)
        frames.append(frame)

    return np.hstack(tuple(frames))[:size]


class GenerateStationaryProcessTestCase(TestCase):
    PSDEstimation = namedtuple("PSDEstimation", ["freq", "s"])

    def estimate_stationary_params(self, x, fs):
        m = np.mean(x)
        std = np.var(x)
        freq, s = signal.welch(x - m, fs=fs, nperseg=256, return_onesided=True, scaling='density')
        return m, std, self.PSDEstimation(freq, s)

    def test_white_noise_1(self):
        self._test_white_noise(fs=0.5)

    def test_white_noise_2(self):
        self._test_white_noise(fs=1.0)

    def test_white_noise_3(self):
        self._test_white_noise(fs=2.0)

    def test_white_noise_4(self):
        self._test_white_noise(fs=5.0)

    def _test_white_noise(self, fs):
        f_psd = lambda f: np.full_like(f, 2 / fs)
        x_iter = generate_process_realization(f_psd=f_psd, fs=fs, window_size=100)

        x = build_sequence(x_iter, 1_000_000)
        x_m, x_var, x_psd = self.estimate_stationary_params(x, fs=fs)

        np.testing.assert_allclose(x_m, 0.0, atol=0.01)
        np.testing.assert_allclose(x_var, 1.0, atol=0.03)

        s_exp = f_psd(x_psd.freq)
        assert_almostclose(x_psd.s, s_exp, atol=0.05, rtol=0.05, max_mismatch=0.05,
                           err_msg='Params: fs = {}'.format(fs))

    def test_stationary_1(self):
        self._test_stationary_process(fs_k=3)

    def test_stationary_2(self):
        self._test_stationary_process(fs_k=8)

    def _test_stationary_process(self, fs_k):
        fm = 2.0
        fs = fm * fs_k
        f_psd_1 = build_norm_fun(0.6 * fm, alpha=0.008)
        f_psd_2 = build_norm_fun(0.4 * fm, alpha=0.005)
        f_psd = lambda f: f_psd_1(f) * 0.6 + f_psd_2(f) * 0.9

        x_iter = generate_process_realization(
            f_psd=f_psd,
            fs=fs
        )

        x = build_sequence(x_iter, 1_000_000)
        x_m, x_var, x_psd = self.estimate_stationary_params(x, fs=fs)

        np.testing.assert_allclose(x_m, 0.0, atol=0.01)
        np.testing.assert_allclose(x_var, 1.5, atol=0.02)

        s_exp = f_psd(x_psd.freq)
        assert_almostclose(x_psd.s, s_exp, rtol=0.05, atol=0.03, max_mismatch=0.075)

    def test_stationary_process_with_m_and_std(self):
        fm = 2.0
        fs = fm * 6
        f_psd_1 = build_norm_fun(0.6 * fm, alpha=0.008)
        f_psd_2 = build_norm_fun(0.4 * fm, alpha=0.005)
        f_psd = lambda f: f_psd_1(f) * 0.6 + f_psd_2(f) * 0.9

        x_iter = generate_process_realization(
            f_psd=f_psd,
            fs=fs,
            f_m=1.5,
            f_std=2
        )

        x = build_sequence(x_iter, 1_000_000)
        x_m, x_var, x_psd = self.estimate_stationary_params(x, fs=fs)

        np.testing.assert_allclose(x_m, 1.5, atol=0.01)
        np.testing.assert_allclose(x_var, 4.0, atol=0.5)

        s_exp = (f_psd(x_psd.freq) / 1.5) * 4

        assert_almostclose(x_psd.s, s_exp, rtol=0.05, atol=0.03, max_mismatch=0.075)


def rolling_mean(x, win_size):
    """
    Calculate rolling mean.

    :param x:
    :param win_size:
    :return:
    """
    x_cumsum = np.cumsum(np.insert(x, 0, 0))
    m = (x_cumsum[win_size:] - x_cumsum[:-win_size]) / win_size
    res = np.empty_like(x)
    right_indent = win_size // 2
    left_indent = win_size - right_indent
    res[:left_indent] = m[0]
    res[left_indent:1 - right_indent] = m
    res[-right_indent:] = m[-1]
    return res


class GenerateNonStationaryWithMeanAndStdProcessTestCase(TestCase):
    PSDEstimation = namedtuple("PSDEstimation", ["freq", "s"])

    def estimate_params(self, x, fs, *, swin_size=64, swin_m_size=None, swin_std_size=None):
        swin_m_size = swin_m_size or swin_size
        swin_std_size = swin_std_size or swin_size

        x_m = rolling_mean(x, swin_m_size)
        x -= x_m

        x_var = rolling_mean(x ** 2, swin_std_size)
        x_std = np.sqrt(x_var)

        x /= x_std

        freq, s = signal.welch(x, fs=fs, nperseg=256, return_onesided=True, scaling='density')
        return x_m, x_std, self.PSDEstimation(freq, s)

    def build_f_psd(self, fs):
        fm = fs / 6
        f_psd_1 = build_norm_fun(0.7 * fm, alpha=0.008)
        f_psd_2 = build_norm_fun(0.3 * fm, alpha=0.005)
        f_psd = lambda f: f_psd_1(f) * 0.3 + f_psd_2(f) * 0.7
        return f_psd

    def test_m_and_std_functions(self):
        # build psd function
        fs = 6.0
        f_psd = self.build_f_psd(fs)

        num_samples = 1_000_000
        t_max = num_samples / fs

        def f_m(t):
            val = np.cos(((t / t_max) - 0.5) * np.pi)
            return val

        def f_std(t):
            val = erf((t / t_max - 0.5) * 6) / 2 + 1
            return val

        x_iter = generate_process_realization(
            f_psd=f_psd,
            fs=fs,
            f_m=f_m,
            f_std=f_std
        )

        x = build_sequence(x_iter, 1_000_000)

        x_m, x_std, x_psd = self.estimate_params(x, fs=fs, swin_m_size=4096, swin_std_size=65536)
        t = np.linspace(0, t_max, num_samples)

        assert_almostclose(x_m, f_m(t), atol=0.05, rtol=0.05, max_mismatch=0.01)
        assert_almostclose(x_std, f_std(t), atol=0.05, rtol=0.05, max_mismatch=0.01)
        assert_almostclose(x_psd.s, f_psd(x_psd.freq), atol=0.05, rtol=0.05, max_mismatch=0.01)

    def test_m_and_std_functions_with_out(self):
        # build psd function
        fs = 6.0
        f_psd = self.build_f_psd(fs)

        num_samples = 1_000_000
        t_max = num_samples / fs

        def f_m(t, out=None):
            out[...] = np.cos(((t / t_max) - 0.5) * np.pi)
            return out

        def f_std(t, out=None):
            out[...] = erf((t / t_max - 0.5) * 6) / 2 + 1
            return out

        x_iter = generate_process_realization(
            f_psd=f_psd,
            fs=fs,
            f_m=f_m,
            f_std=f_std
        )

        x = build_sequence(x_iter, 1_000_000)

        x_m, x_std, x_psd = self.estimate_params(x, fs=fs, swin_m_size=4096, swin_std_size=65536)
        t = np.linspace(0, t_max, num_samples)

        assert_almostclose(x_m, f_m(t, np.empty_like(t)), atol=0.05, rtol=0.05, max_mismatch=0.01)
        assert_almostclose(x_std, f_std(t, np.empty_like(t)), atol=0.05, rtol=0.05, max_mismatch=0.01)
        assert_almostclose(x_psd.s, f_psd(x_psd.freq), atol=0.05, rtol=0.05, max_mismatch=0.01)


class GenerateNonStationaryProcessTestCase(TestCase):
    PSDEstimation = namedtuple("PSDEstimation", ["freq", "s"])

    def build_smooth_transition_fun(self, t_b, x_b, t_e, x_e):
        t_m = (t_b + t_e) / 2
        w = np.pi / (2 * (t_e - t_b))

        def transition_fun(t):
            phi = (t - t_m) * w * 2
            if np.isscalar(t):
                phi = np.pi / 2 if phi >= np.pi / 2 else phi
                phi = -np.pi / 2 if phi <= -np.pi / 2 else phi
            else:
                phi[phi >= np.pi / 2] = np.pi / 2
                phi[phi <= -np.pi / 2] = -np.pi / 2
            res = (np.sin(phi) + 1) / 2
            res *= (x_e - x_b)
            res += x_b
            return res

        return transition_fun

    def estimate_params(self, x_ensemble, fs):
        """
        Estimate non stationary process parameters from ensemble.

        :param x_ensemble: signal ensemble. First axis - t, seconds - realizations
        :param fs:
        :return:
        """
        x_m = np.mean(x_ensemble, axis=-1)
        x_ensemble -= x_m[..., np.newaxis]

        x_std = np.std(x_ensemble, axis=-1)
        x_ensemble /= x_std[..., np.newaxis]

        # create ensemble view with the following axes:
        # 0 - time of the window
        # 1 - samples within window
        # 2 - ensemble
        win_size = 128
        freq = np.fft.rfftfreq(win_size, d=1 / fs)
        x = np.lib.stride_tricks.as_strided(
            x_ensemble,
            shape=(x_ensemble.shape[0] - win_size + 1, win_size, x_ensemble.shape[1]),
            strides=(x_ensemble.strides[0],) + x_ensemble.strides,
            writeable=False
        )
        s_res = np.empty((x_ensemble.shape[0], len(freq)))
        s_start_i = win_size // 2
        s_end_i = s_start_i + x.shape[0]

        def calc_periodogram(x_ensemble_t):
            _, s = signal.periodogram(x_ensemble_t, fs=fs, return_onesided=True, scaling='density', axis=0)
            s = s.mean(-1)
            return s

        with mp.Pool() as pool:
            for i, s in enumerate(pool.imap(calc_periodogram, x, chunksize=10), start=s_start_i):
                s_res[i] = s

        s_res[:s_start_i] = s_res[s_start_i]
        s_res[s_end_i:] = s_res[s_end_i - 1]

        return x_m, x_std, self.PSDEstimation(freq, s_res)

    def build_f_psd(self, fs, t_max):
        fm = fs / 6

        f_f_0_1 = self.build_smooth_transition_fun(0, 0.7 * fm, t_max, 0.4 * fm)
        f_f_0_2 = self.build_smooth_transition_fun(0, 0.3 * fm, t_max, 0.8 * fm)
        f_alpha_1 = self.build_smooth_transition_fun(0, 0.008, t_max, 0.009)
        f_alpha_2 = self.build_smooth_transition_fun(0, 0.005, t_max, 0.004)

        def f_psd(f, t):
            f_0_1 = f_f_0_1(t)
            f_0_2 = f_f_0_2(t)
            alpha_1 = f_alpha_1(t)
            alpha_2 = f_alpha_2(t)
            return f_psd_norm(f, f_0=f_0_1, alpha=alpha_1) * 0.3 + f_psd_norm(f, f_0=f_0_2, alpha=alpha_2) * 0.7

        return f_psd

    def test_ensemble(self):
        fs = 6.0
        num_samples = 8192
        window_size = 256
        num_realization = 1000
        t = np.arange(num_samples) / fs
        t_max = t[-1]

        f_m = self.build_smooth_transition_fun(0, -1, t_max, 2)
        f_std = self.build_smooth_transition_fun(0, 2, t_max, 1.5)
        f_psd = self.build_f_psd(fs, t_max)

        def create_realization(*args, **kwargs):
            x_iter = generate_process_realization(
                f_psd=f_psd,
                fs=fs,
                f_m=f_m,
                f_std=f_std,
                window_size=window_size
            )
            return build_sequence(x_iter, num_samples)

        x_ensemble = np.empty((num_samples, num_realization), dtype=np.float64)
        with mp.Pool() as pool:
            for i, x in enumerate(pool.imap(create_realization, range(num_realization), chunksize=10)):
                x_ensemble[:, i] = x

        x_m, x_std, x_psd = self.estimate_params(x_ensemble, fs)

        assert_almostclose(x_m, f_m(t), atol=0.2, rtol=0.05, max_mismatch=0.01)
        assert_almostclose(x_std, f_std(t), atol=0.2, rtol=0.05, max_mismatch=0.01)

        s_expected = np.empty_like(x_psd.s)
        for i, t_i in enumerate(t):
            s_expected[i] = f_psd(x_psd.freq, t_i)

        assert_almostclose(x_psd.s, s_expected, atol=0.2, rtol=0.05, max_mismatch=0.01)
