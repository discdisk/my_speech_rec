import chainer.functions as F
import chainer.links as L
import chainer
import numpy as np
from chainer.backends import cuda


class NstepLSTM_normalization(chainer.ChainList):
    """docstring for NstepLSTM_normalization"""

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        super(NstepLSTM_normalization, self).__init__()
        with self.init_scope():
            for i in range(n_layers):
                self.add_link( LSTM_normalization(in_size, out_size, dropout))

    def __call__(self, xs):
        for child in self.children():
            xs = child(xs)
        return xs

class LSTM_normalization(chainer.Chain):
    """docstring for LSTM_normalization"""
    def __init__(self, in_size, out_size, dropout, layer_norm_on=True):
        super(LSTM_normalization, self).__init__()
        initializer = chainer.initializers.Normal()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1, in_size, out_size, 0)
            self.layer_norm = L.LayerNormalization(out_size,eps=1e-5, initial_gamma=initializer)
            self.layer_norm_on = layer_norm_on
            self.dropout = dropout

    def __call__(self,xs):
        _, _, ys = self.lstm(None, None, xs)
        return [F.dropout(self.layer_norm(y),self.dropout) for y in ys]
        # return ys




class Additive_Attention(chainer.Chain):
    """docstring for Attention"""

    def __init__(self, n_mid_units, win_size, batch_size, att_units_size, frame_level = True):
        super(Attention, self).__init__()

        self.win_size = win_size
        self.batch_size = batch_size
        self.att_units_size = att_units_size
        self.att_out_size = 1 if frame_level else n_mid_units

        initializer = chainer.initializers.LeCunNormal()

        with self.init_scope():
            self.last_out     = L.Linear(None, self.att_units_size, initialW=initializer)
            self.hidden_layer = L.Linear(n_mid_units, self.att_units_size, initialW=initializer)
            self.att_cal      = L.Linear(self.att_units_size, self.att_out_size, initialW=initializer)

    def __call__(self, gt, last, mask):


        hx = self.hidden_layer(gt, n_batch_axes=2)

        Z = self.last_out(last)
        Z = F.broadcast_to(Z, (self.win_size, self.batch_size, self.att_units_size))

        attend = hx + Z

        attend =self.att_cal(
                F.tanh(attend.reshape(-1, self.att_units_size))
            ).reshape(-1, self.batch_size, self.att_out_size) + mask


        attend = F.softmax(attend,axis=0)


        context = attend * gt

        context = F.sum(context, axis=0, keepdims=False) * self.win_size

        return context
