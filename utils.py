import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer.dataset import to_device
from chainer.dataset import concat_examples
from chainer import training
import numpy as np
import decoder
from chainer.functions.loss.ctc import ConnectionistTemporalClassification
from chainer.utils import collections_abc
from chainer.utils import type_check
from chainer.iterators.order_samplers import OrderSampler
import math
class util():
    """docstring for utils"""
    def __init__(self,device,blank):
        super(util, self).__init__()
        self.xp = cuda.cupy if device>-1 else np
        self.device=device
        self.blank=blank
        self.stacked_frames=8
        self.skip_size=3
        

    def converter(self,batch,device=-1):
        # alternative to chainer.dataset.concat_examples
        DATA_SHAPE=40 #40 log filterbank

        Xs = [self.xp.asarray(list(X)).astype(self.xp.float32).reshape(-1,DATA_SHAPE)for X, _ in batch]
        Xs =[F.concat((X,self.xp.zeros((self.stacked_frames-len(X),DATA_SHAPE),dtype=self.xp.float32)),axis=0) if len(X)<self.stacked_frames else X for X in Xs]
        Xs =[F.pad_sequence([X[i:i+self.stacked_frames] for i in range(0,len(X),self.skip_size)]).reshape(-1,DATA_SHAPE*self.stacked_frames) for X in Xs]
        self.numframes = [len(X) for X in Xs]


        word_label = [self.xp.asarray(lab[0]).astype(self.xp.int32) for _, lab in batch]
        type_label = [self.xp.asarray(lab[1]).astype(self.xp.int32) for _, lab in batch]
        self.label_length=[len(l) for l in word_label]



        self.label_length=self.xp.asarray(self.label_length,dtype=self.xp.int32)

        lable_batch=(word_label,type_label)

        return Xs, lable_batch


    def ctc_loss(self,ys, lable_batch):
        (word_label,type_label)=lable_batch
        (word_ys,type_ys)=ys

        lables = concat_examples(word_label, self.device, padding=self.blank)
        type_labels = concat_examples(type_label, self.device, padding=self.blank)
        numframes=self.numframes

        input_length = self.xp.asarray(numframes, dtype=self.xp.int32)


        word_loss = F.connectionist_temporal_classification(word_ys, lables, self.blank, input_length, self.label_length)
        type_loss = F.connectionist_temporal_classification(type_ys, type_labels, self.blank, input_length, self.label_length)
        loss=(4*word_loss+type_loss)/5

        print(loss)
        # print(math.isnan(loss.data))
        # if loss.data>1000 or math.isnan(loss.data):
        #     print(list(zip(input_length, self.label_length)))

        return loss
    

    def cal_WER(self,x,y,predictor):
        result=predictor(x)
        result=F.stack(result,axis=1)
        out=[]
        WER=0
        for i,res in enumerate(result):
            out.append(res[:self.numframes[i]])
        for r,h in zip(y,out):
            WER+=self._wer(r.data,decoder.decode(h.data)[0])



        return WER/len(y)


    def _wer(self,r, h):
        d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
        d = d.reshape((len(r)+1, len(h)+1))
        for i in range(len(r)+1):
            for j in range(len(h)+1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion    = d[i][j-1] + 1
                    deletion     = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(r)][len(h)],len(h)




class LengthOrderSampler(OrderSampler):

    """Sampler that generates lenth orders.

    This is expected to be used together with Chainer's iterators.
    An order sampler is called by an iterator every epoch.


    """

    def __init__(self, dataset, maxlen_in=800, maxlen_out=150,length_order=None):
        if length_order is None:
            print('ordering')
            self._data = dataset
            order=sorted(range(len(self._data)),key=lambda k:len(self._data[k][0]))
            print('orderin ===>sorted')
            self.length_order=[]
            print('orderin ===>picking')
            for o in order:
                data=self._data[o]
                if len(data[0])<maxlen_in and len(data[1])<maxlen_out:
                    self.length_order.append(o)
            print('ordering done')
        else:
            self.length_order=length_order
            print('loaded')

    def __call__(self, current_order, current_position):
        return self.length_order



import collections,six,logging
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


if __name__ == '__main__':
    u=util(0,0)
    print(u._wer([1,2,2,4,2,3,4],[0,4,2,3,4]))