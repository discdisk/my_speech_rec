import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer.dataset import to_device
import numpy as np
from chainer.utils import collections_abc
from chainer.utils import type_check


import collections
import six
import logging
import math


def sum_sqnorm(arr):
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device_from_array(x) as dev:
            if x is not None:
                x = x.ravel()
                s = x.dot(x)
                sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])


class myOptimizers(chainer.optimizers.RMSprop):
    """docstring for myUpdater"""

    def update(self, lossfun=None, *args, **kwds):
        """Updates parameters based on a loss function or computed gradients.

        This method runs in two ways.

        - If ``lossfun`` is given, then it is used as a loss function to
          compute gradients.
        - Otherwise, this method assumes that the gradients are already
          computed.

        In both cases, the computed gradients are used to update parameters.
        The actual update routines are defined by the update rule of each
        parameter.

        """
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)

            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()

            loss.backward(loss_scale=self._loss_scale)
            loss.unchain_backward()
            del loss

        grad_norm = np.sqrt(sum_sqnorm(
            [p.grad for p in self.target.params(False)]))

        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:

            self.reallocate_cleared_grads()

            self.call_hooks('pre')

            self.t += 1

            for param in self.target.params():
                param.update()

            self.reallocate_cleared_grads()

            self.call_hooks('post')


class myAdamOptimizers(chainer.optimizers.Adam):
    """docstring for myUpdater"""

    def update(self, lossfun=None, *args, **kwds):
        """Updates parameters based on a loss function or computed gradients.

        This method runs in two ways.

        - If ``lossfun`` is given, then it is used as a loss function to
          compute gradients.
        - Otherwise, this method assumes that the gradients are already
          computed.

        In both cases, the computed gradients are used to update parameters.
        The actual update routines are defined by the update rule of each
        parameter.

        """
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)

            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()

            loss.backward(loss_scale=self._loss_scale)
            loss.unchain_backward()
            del loss

        grad_norm = np.sqrt(sum_sqnorm(
            [p.grad for p in self.target.params(False)]))

        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:

            self.reallocate_cleared_grads()

            self.call_hooks('pre')

            self.t += 1

            for param in self.target.params():
                param.update()

            self.reallocate_cleared_grads()

            self.call_hooks('post')
