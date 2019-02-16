import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer.training import extensions
from chainer import reporter as reporter_module
from chainer import function
import numpy as np

class EEvaluator(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']
        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        while True:
            batch=it.next()
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            summary.add(observation)
            if it.is_new_epoch:
                break
        out=summary.compute_mean()
        print('#############################################',out)
        return out