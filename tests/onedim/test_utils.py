from unittest import TestCase

import numpy as np
from hamcrest import assert_that, instance_of, not_

from vznncv.signal.generator.onedim._utils import UnaryFunction, BinaryFunction, TrendFunction


class TrendFunctionTestCase(TestCase):
    def test_default(self):
        fun = TrendFunction(fun=None, default_value=2.0)
        t = np.linspace(0, 5, 10)
        out = np.zeros_like(t)

        fun.apply_fun(t, out=out)
        np.allclose(out, 2.0)

    def test_constant(self):
        fun = TrendFunction(fun=2.0)
        t = np.linspace(0, 5, 10)
        out = np.zeros_like(t)

        fun.apply_fun(t, out=out)
        np.allclose(out, 2.0)

    def test_fun_without_out(self):
        def trend_fun(t):
            return np.zeros_like(t) + 2.0

        fun = TrendFunction(fun=trend_fun)
        t = np.linspace(0, 5, 10)
        out = np.zeros_like(t)

        fun.apply_fun(t, out=out)
        np.allclose(out, 2.0)

    def test_fun_with_out(self):
        def trend_fun(t, out=None):
            out.fill(2)

        fun = TrendFunction(fun=trend_fun)
        t = np.linspace(0, 5, 10)
        out = np.zeros_like(t)

        fun.apply_fun(t, out=out)
        np.allclose(out, 2.0)

    def test_optional_args(self):
        def trend_fun(t, out=None, param_1=0, param_2=2):
            out.fill(2)

        fun = TrendFunction(fun=trend_fun)
        t = np.linspace(0, 5, 10)
        out = np.zeros_like(t)

        fun.apply_fun(t, out=out)
        np.allclose(out, 2.0)

    def test_invalid_args(self):
        def trend_fun(t, x, out=None, param_1=0):
            out.fill(2)

        with self.assertRaises(Exception):
            fun = TrendFunction(fun=trend_fun)
            t = np.linspace(0, 5, 10)
            out = np.zeros_like(t)
            fun.apply_fun(t, out=out)


class FunctionTestCase(TestCase):
    def test_funtions(self):
        def fun_0_arg():
            pass

        def fun_1_arg(a):
            pass

        def fun_1_arg_kwarg(a, b=None):
            pass

        def fun_2_arg(a, b):
            pass

        def fun_2_arg_kwarg(a, b, c=None):
            pass

        def fun_3_arg(a, b, c):
            pass

        assert_that(fun_0_arg, not_(instance_of(UnaryFunction)))
        assert_that(fun_1_arg, instance_of(UnaryFunction))
        assert_that(fun_1_arg_kwarg, instance_of(UnaryFunction))
        assert_that(fun_2_arg, not_(instance_of(UnaryFunction)))
        assert_that(fun_2_arg_kwarg, not_(instance_of(UnaryFunction)))
        assert_that(fun_3_arg, not_(instance_of(UnaryFunction)))

        assert_that(fun_0_arg, not_(instance_of(BinaryFunction)))
        assert_that(fun_1_arg, not_(instance_of(BinaryFunction)))
        assert_that(fun_1_arg_kwarg, not_(instance_of(BinaryFunction)))
        assert_that(fun_2_arg, instance_of(BinaryFunction))
        assert_that(fun_2_arg_kwarg, instance_of(BinaryFunction))
        assert_that(fun_3_arg, not_(instance_of(BinaryFunction)))

    def test_class_method(self):
        class A:
            @classmethod
            def fun_0_arg(cls):
                pass

            @classmethod
            def fun_1_arg(cls, a):
                pass

            @classmethod
            def fun_1_arg_kwarg(cls, a, b=None):
                pass

            @classmethod
            def fun_2_arg(cls, a, b):
                pass

            @classmethod
            def fun_2_arg_kwarg(cls, a, b, c=None):
                pass

            @classmethod
            def fun_3_arg(cls, a, b, c):
                pass

        assert_that(A.fun_0_arg, not_(instance_of(UnaryFunction)))
        assert_that(A.fun_1_arg, instance_of(UnaryFunction))
        assert_that(A.fun_1_arg_kwarg, instance_of(UnaryFunction))
        assert_that(A.fun_2_arg, not_(instance_of(UnaryFunction)))
        assert_that(A.fun_2_arg_kwarg, not_(instance_of(UnaryFunction)))
        assert_that(A.fun_3_arg, not_(instance_of(UnaryFunction)))

        assert_that(A.fun_0_arg, not_(instance_of(BinaryFunction)))
        assert_that(A.fun_1_arg, not_(instance_of(BinaryFunction)))
        assert_that(A.fun_1_arg_kwarg, not_(instance_of(BinaryFunction)))
        assert_that(A.fun_2_arg, instance_of(BinaryFunction))
        assert_that(A.fun_2_arg_kwarg, instance_of(BinaryFunction))
        assert_that(A.fun_3_arg, not_(instance_of(BinaryFunction)))

    def test_method(self):
        class A:
            def fun_0_arg(self):
                pass

            def fun_1_arg(self, a):
                pass

            def fun_1_arg_kwarg(self, a, b=None):
                pass

            def fun_2_arg(self, a, b):
                pass

            def fun_2_arg_kwarg(self, a, b, c=None):
                pass

            def fun_3_arg(self, a, b, c):
                pass

        a = A()

        assert_that(a.fun_0_arg, not_(instance_of(UnaryFunction)))
        assert_that(a.fun_1_arg, instance_of(UnaryFunction))
        assert_that(a.fun_1_arg_kwarg, instance_of(UnaryFunction))
        assert_that(a.fun_2_arg, not_(instance_of(UnaryFunction)))
        assert_that(a.fun_2_arg_kwarg, not_(instance_of(UnaryFunction)))
        assert_that(a.fun_3_arg, not_(instance_of(UnaryFunction)))

        assert_that(a.fun_0_arg, not_(instance_of(BinaryFunction)))
        assert_that(a.fun_1_arg, not_(instance_of(BinaryFunction)))
        assert_that(a.fun_1_arg_kwarg, not_(instance_of(BinaryFunction)))
        assert_that(a.fun_2_arg, instance_of(BinaryFunction))
        assert_that(a.fun_2_arg_kwarg, instance_of(BinaryFunction))
        assert_that(a.fun_3_arg, not_(instance_of(BinaryFunction)))

    def test_callable(self):
        class A0Arg:
            def __call__(self):
                pass

        class A1Arg:
            def __call__(self, a):
                pass

        class A2Arg:
            def __call__(self, a, b):
                pass

        class A3Arg:
            def __call__(self, a, b, c):
                pass

        assert_that(A0Arg(), not_(instance_of(UnaryFunction)))
        assert_that(A1Arg(), instance_of(UnaryFunction))
        assert_that(A2Arg(), not_(instance_of(UnaryFunction)))
        assert_that(A3Arg(), not_(instance_of(UnaryFunction)))

        assert_that(A0Arg(), not_(instance_of(BinaryFunction)))
        assert_that(A1Arg(), not_(instance_of(BinaryFunction)))
        assert_that(A2Arg(), instance_of(BinaryFunction))
        assert_that(A3Arg(), not_(instance_of(BinaryFunction)))
