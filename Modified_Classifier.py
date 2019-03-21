from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
import chainer

class MYClassifier(chainer.links.Classifier):


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
        word_loss,char_loss = self.lossfun(self.y, t)


        self.loss=(7*word_loss+3*char_loss)/10


        if -200<self.loss.data <2000:
            reporter.report({'loss': self.loss}, self)
            reporter.report({'word_loss': word_loss}, self)
            reporter.report({'char_loss': char_loss}, self)
        elif self.loss.data>0:
            reporter.report({'loss': 2000}, self)
        else:
            reporter.report({'loss': -200}, self)

            
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss