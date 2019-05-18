import logging

import numpy as np

logger = logging.getLogger(__name__)


def assert_almostclose(actual, desired, *, rtol=1e-7, atol=0.0, equal_nan=True,
                       err_msg='', verbose=True, max_mismatch=0.0):
    """
    Analog of the :func:`numpy.testing.assert_allclose` that has a ``max_mismatch`` parameter
    that specify maximum part of the outliers.

    :param actual:
    :param desired:
    :param rtol:
    :param atol:
    :param equal_nan:
    :param err_msg:
    :param verbose:
    :param max_mismatch:
    :return:
    """
    actual, desired = np.asanyarray(actual), np.asanyarray(desired)
    cmp_res = np.isclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan)

    mismatch_k = 1 - cmp_res.sum() / cmp_res.size
    logger.debug('Mismatch: {:.2%}'.format(mismatch_k))
    if mismatch_k >= max_mismatch:
        if err_msg:
            err_msg = '{}\n'.format(err_msg)
        raise AssertionError("{}Mismatch: {:.01%}\nactural: {}\ndesired: {}"
                             .format(err_msg, mismatch_k, actual, desired))
