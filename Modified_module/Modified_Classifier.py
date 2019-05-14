from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
import chainer
import chainer.functions as F
import numpy as np
from chainer.dataset import to_device


def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = 0
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank
    new_labels = [l for l in new_labels if l != blank]

    return np.array(new_labels)


def _wer(r, h):
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / len(h)


class MYClassifier(chainer.links.Classifier):

    def set_utils(self, utils):
        self.batch_size = self.predictor.batch_size
        self.utils = utils

    def forward(self, *args, **kwargs):

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y, t)

        if -1000 < self.loss.data/self.batch_size < 1000:
            reporter.report({'loss': self.loss/self.batch_size}, self)
            # reporter.report({'char_loss': char_loss}, self)
        else:
            print('loss fucked up!!!!!!!!!!!!!')
            

        if self.compute_accuracy:
            wer = 0
            ys = [y.data[:n] for y, n in zip(F.stack(self.y[0], 1), self.y[1])]

            for yy, tt in zip(ys, to_device(-1, t[0])):
                out = remove_blank(F.argmax(yy, axis=1).data)
                out = [int(o) for o in out]

            
                wer += _wer(out, tt)


            reporter.report({'accuracy': wer / len(ys)}, self)
        return self.loss

if __name__ == '__main__':
   print( _wer([12, 3828],[  12, 1546]))