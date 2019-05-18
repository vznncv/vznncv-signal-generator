import re
from unittest import TestCase

import numpy as np
from hamcrest import assert_that, calling, raises

from vznncv.signal.generator.onedim._window import \
    calculate_correlation_interval, \
    build_f_fun_from_f_psd, \
    calculate_correlation_interval_for_psd


class CalculateCorrelationIntervalTestCase(TestCase):
    def test_fs_scale(self):
        f_r = lambda t: np.sinc(t * 0.1)
        alpha = 0.01
        fs = 1.0
        t_k_exp = 315.4

        for k in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            t_k = calculate_correlation_interval(f_r, alpha=alpha, fs=fs * k)
            np.testing.assert_allclose(t_k, t_k_exp, rtol=0.01)

    def test_diff_alpha(self):
        f_r = lambda t: np.sinc(t * 0.1)
        alpha_vals = [0.001, 0.01, 0.05, 0.1]
        t_k_exp_val = [3175, 315.4, 56.53, 26.8]
        fs = 1.0

        for alpha, t_k_exp in zip(alpha_vals, t_k_exp_val):
            t_k = calculate_correlation_interval(f_r, alpha=alpha, fs=fs)
            np.testing.assert_allclose(t_k, t_k_exp, rtol=0.01)

    def test_t_k_not_found(self):
        f_r = lambda t: np.sin(t * 0.1)
        r_min = 0.01
        fs = 1.0

        assert_that(
            calling(calculate_correlation_interval).with_args(f_r, r_min, fs=fs),
            raises(Exception, pattern=re.compile(r'\bcannot find\b', re.IGNORECASE))
        )


class BuildAutocorrFunFromPSDFunTestCase(TestCase):
    def build_rc_f_psd(self, f_0, alpha):
        def f_psd(f):
            return alpha / (alpha ** 2 + ((f - f_0) * 2 * np.pi) ** 2) + \
                   alpha / (alpha ** 2 + ((f + f_0) * 2 * np.pi) ** 2)

        return f_psd

    def build_rc_f_r(self, f_0, alpha):
        def f_r(t):
            return np.exp(-alpha * np.abs(t)) * np.cos(f_0 * np.pi * 2 * t)

        return f_r

    def test_fun_1(self):
        f_0 = 0.15
        alpha = 0.5

        f_psd = self.build_rc_f_psd(f_0=f_0, alpha=alpha)
        f_r_exp = self.build_rc_f_r(f_0=f_0, alpha=alpha)

        f_r = build_f_fun_from_f_psd(f_psd)
        t = np.linspace(-50, 50, 1000)

        r = f_r(t)
        r_exp = f_r_exp(t)

        np.testing.assert_allclose(r, r_exp, rtol=0.05, atol=0.1)

    def test_fun_2(self):
        f_0 = 0.4
        alpha = 0.04
        f_psd = self.build_rc_f_psd(f_0=f_0, alpha=alpha)
        f_r_exp = self.build_rc_f_r(f_0=f_0, alpha=alpha)

        t = np.linspace(-200, 200, 2000)
        f_r = build_f_fun_from_f_psd(f_psd)
        r = f_r(t)
        r_exp = f_r_exp(t)
        np.testing.assert_allclose(r, r_exp, rtol=0.02, atol=0.01)

    def test_fun_3(self):
        f_0 = 0.05
        alpha = 0.05
        f_psd = self.build_rc_f_psd(f_0=f_0, alpha=alpha)
        f_r_exp = self.build_rc_f_r(f_0=f_0, alpha=alpha)

        t = np.linspace(-100, 100, 1000)
        f_r = build_f_fun_from_f_psd(f_psd)
        r = f_r(t)
        r_exp = f_r_exp(t)
        np.testing.assert_allclose(r, r_exp, rtol=0.02, atol=0.01)

    def test_diff_fs(self):
        f_0 = 0.05
        alpha = 0.05
        f_psd = self.build_rc_f_psd(f_0=f_0, alpha=alpha)
        f_r_exp = self.build_rc_f_r(f_0=f_0, alpha=alpha)

        t = np.linspace(-100, 100, 1000)
        r_exp = f_r_exp(t)
        for fs in [0.4, 0.7, 1.0, 2.0, 5.0, 10.0]:
            f_r = build_f_fun_from_f_psd(f_psd, fs=fs)
            r = f_r(t)
            np.testing.assert_allclose(r, r_exp, rtol=0.02, atol=0.01)


class CalculateCorrelationIntervalForPSDTestCase(TestCase):
    def build_rc_f_psd(self, f_0, alpha):
        def f_psd(f):
            return alpha / (alpha ** 2 + ((f - f_0) * 2 * np.pi) ** 2) + \
                   alpha / (alpha ** 2 + ((f + f_0) * 2 * np.pi) ** 2)

        return f_psd

    def test_fs_scale(self):
        f_r = self.build_rc_f_psd(0.1, 0.05)
        alpha = 0.01
        fs = 1.0
        t_k_exp = 90.6

        for k in [0.5, 1.0, 2.0, 5.0, 10.0]:
            t_k = calculate_correlation_interval_for_psd(f_r, alpha=alpha, fs=fs * k)
            np.testing.assert_allclose(t_k, t_k_exp, rtol=0.01)

