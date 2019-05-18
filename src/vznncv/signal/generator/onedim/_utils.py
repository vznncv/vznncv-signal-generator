import inspect

import numpy as np


def get_random_state(value) -> np.random.RandomState:
    """
    Get random state generator.

    :param value: :class:`numpy.random.RandomState` object or seed for it
    :return: ` np.random.RandomState` object
    """
    if hasattr(value, 'randn'):
        pass
    elif value is None:
        value = np.random.RandomState()
    else:
        value = np.random.RandomState(value)
    return value


class TrendFunction:
    """
    Helper wrapper/validator for a expected value/standard deviation functions.
    """
    _POS_PARAM_KIND = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def __init__(self, fun, default_value=None):
        self.original_fun = fun
        self.default_value = default_value

        if fun is None:
            if default_value is None:
                raise ValueError("function and default values aren't set")
            self.apply_fun = lambda t, out: out.fill(default_value)
        elif np.isscalar(fun):
            self.apply_fun = lambda t, out: out.fill(fun)
        elif callable(fun):
            parameter = inspect.signature(fun).parameters

            parameter = {k: v for k, v in parameter.items() if v.kind in _NArgFunctionMeta._POS_PARAM_KIND}
            out_param = parameter.pop('out', None)
            num_pos_args = sum(1 for p in parameter.values() if p.default is inspect.Parameter.empty)

            if num_pos_args != 1:
                raise ValueError("'{}' isn't unary function".format(fun))

            if out_param is None:
                def apply_fun(t, out):
                    out[...] = fun(t)

                self.apply_fun = apply_fun
            else:
                self.apply_fun = fun
        else:
            raise ValueError("'{}' isn't None, scalar or function".format(fun))

    def __call__(self, t, out=None):
        if out is None:
            out = np.empty_like(t)
        self.apply_fun(t, out=out)
        return out


class _NArgFunctionMeta(type):
    _POS_PARAM_KIND = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def __instancecheck__(self, instance):
        if not callable(instance):
            return False

        parameter = inspect.signature(instance).parameters
        num_pos_args = sum(
            1 for p in parameter.values()
            if p.kind in _NArgFunctionMeta._POS_PARAM_KIND
            and p.default is inspect.Parameter.empty
        )

        return num_pos_args == self.NARGS


class UnaryFunction(metaclass=_NArgFunctionMeta):
    NARGS = 1


class BinaryFunction(metaclass=_NArgFunctionMeta):
    NARGS = 2
