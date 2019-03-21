
import chainer.functions as F
import chainer.links as L
import chainer
import numpy as np
from chainer.backends import cuda


class Linear3D(L.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear3D, self).__init__(*args, **kwargs)

    def call(self, x):
        return super(Linear3D, self).__call__(x)

    def __call__(self, x):
        if x.ndim == 2:
            return self.call(x)
        assert x.ndim == 3

        x_2d = x.reshape((-1, x.shape[-1]))
        out_2d = self.call(x_2d)
        out_3d = out_2d.reshape(x.shape[:-1] + (out_2d.shape[-1], ))
        # (B, S, W)
        return out_3d


class RNN(chainer.Chain):

    def __init__(self, n_lstm_layers, n_mid_units, n_out, win_size, batch_size, att_units_size, dropout=0.5 ):
        super(RNN, self).__init__()

        initializer = chainer.initializers.Normal()
        n_word_out=n_out[0]
        n_char_out=n_out[1]
        with self.init_scope():
            # self.layer_norm1=L.LayerNormalization()
            self.l1 = L.Linear(None, n_mid_units,initialW=initializer)
            # self.batch_norm1=L.BatchNormalization(n_mid_units)
            self.char_out = L.Linear(n_mid_units, n_char_out,initialW=initializer)

            self.lstm1 = L.NStepLSTM(1,
                                    in_size=n_mid_units, \
                                    out_size=n_mid_units,
                                    dropout=dropout,)
            self.lstm2 = L.NStepLSTM(n_lstm_layers,
                                    in_size=n_mid_units, \
                                    out_size=n_mid_units,
                                    dropout=dropout,)
            # self.layer_norm2=L.LayerNormalization()
            # self.batch_norm2=L.BatchNormalization(n_mid_units)





            self.attend = Attention(n_mid_units, n_word_out, win_size, batch_size, att_units_size)



    def __call__(self, x):
        # forward calculation

        h1 = [ F.relu(self.l1(X)) for X in x ]
        _, _, ys = self.lstm1(None, None, h1)

        result_char = [self.char_out(y) for y in ys ]
        result_char=F.pad_sequence(result_char)
        result_char=list(F.stack(result_char,axis=1))

        _, _, ys = self.lstm2(None, None, ys)

        result=self.attend(ys)
        # self.parameter_check()
        return result,result_char







class Attention(chainer.Chain):
    """docstring for Attention"""
    def __init__(self, n_mid_units, n_out, win_size, batch_size, att_units_size, device=-1):
        super(Attention,self).__init__()

        self.pad_size=int((win_size-1)/2)
        self.win_size=win_size
        self.batch_size=batch_size
        self.att_units_size=att_units_size
        self.n_mid_units=n_mid_units
        self.n_out=n_out
        xp=cuda.cupy

        self.Zu = xp.zeros((batch_size,n_mid_units),dtype=np.float32)
        self.pad_zero=xp.zeros((self.pad_size,n_mid_units),dtype=self.xp.float32)

        initializer = chainer.initializers.LeCunNormal()

        with self.init_scope():
            self.last_out = L.Linear(n_mid_units, self.att_units_size,initialW=initializer)
            self.hidden_layer = L.Linear(n_mid_units, self.att_units_size,initialW=initializer)
            self.att_cal = L.Linear(self.att_units_size,n_mid_units,initialW=initializer)
            self.output = L.Linear(n_mid_units, n_out,initialW=initializer)

    def __call__(self, x):
        self.Zu=self.xp.zeros(self.Zu.shape,dtype=np.float32)

        gts=[F.concat((F.concat((self.pad_zero,gt),axis=0),self.pad_zero),axis=0) for gt in x]
        gts=F.pad_sequence(gts,padding=self.xp.zeros((self.n_mid_units),dtype=self.xp.float32))
        gts=F.stack(gts,axis=1)

        result=[]        
        for i in range(len(gts)-self.pad_size*2):
            gt=gts[i:i+self.win_size]
            hx=self.hidden_layer(gt,n_batch_axes=2)

            Z=self.last_out(self.Zu)
            Z=F.broadcast_to(Z,(self.win_size,self.batch_size,self.att_units_size))

            attend=hx+Z

            attend=F.softmax(
                self.att_cal(
                    F.tanh(attend.reshape(-1,self.att_units_size))
                    ).reshape(-1,self.batch_size,self.n_mid_units)
                ,axis=0)

            context=attend*gt


            context=F.sum(context,axis=0,keepdims=False)*self.win_size
            result.append(self.output(context))

            self.Zu=context

        return result


