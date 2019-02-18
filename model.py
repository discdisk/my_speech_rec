
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

def argsort_list_descent(lst):
    return np.argsort([-len(x) for x in lst]).astype('i')


def permutate_list(lst, indices, inv):
    ret = [None] * len(lst)
    if inv:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


class RNN(chainer.Chain):

    def __init__(self, n_lstm_layers, n_mid_units, n_out, win_size, batch_size, att_units_size, dropout=0.5 ):
        super(RNN, self).__init__()

        initializer = chainer.initializers.LeCunNormal()
        with self.init_scope():
            # self.layer_norm1=L.LayerNormalization()
            self.l1 = L.Linear(None, n_mid_units,initialW=initializer)
            # self.batch_norm1=L.BatchNormalization(n_mid_units)

            # self.l2 = L.NStepLSTM(n_lstm_layers,
            #                         in_size=n_mid_units, \
            #                         out_size=n_mid_units,
            #                         dropout=dropout,)

            self.lstm=L.LSTM(in_size=n_mid_units, out_size=n_mid_units).repeat(3)
            # self.layer_norm2=L.LayerNormalization()
            # self.batch_norm2=L.BatchNormalization(n_mid_units)

            self.word_output = L.Linear(n_mid_units, n_out,initialW=initializer)
            self.type_output = L.Linear(n_mid_units, 16,initialW=initializer)





            self.attend = Attention(n_mid_units, n_out, win_size, batch_size, att_units_size)



    def __call__(self, xs):
        # forward calculation

        indices = argsort_list_descent(xs)

        xs = permutate_list(xs, indices, inv=False)

        # h = [ F.relu(self.l1(x)) for x in xs ]

        # _, _, ys = self.l2(None, None, h)

        xs=F.transpose_sequence(xs)

        for x in xs:
            h=F.relu(self.l1(x))

            for child in self.lstm.children():
                h=F.dropout(child(h))
            self.input_query.append(ys)
            context=self.attend(self.input_query,last_context)

            result=self.word_output(context)
            result_type=self.type_output(context)
        # self.parameter_check()
        return result







class Attention(chainer.Chain):
    """docstring for Attention"""
    def __init__(self, n_mid_units, n_out, win_size, batch_size, att_units_size, device=-1):
        super(Attention,self).__init__()


        initializer = chainer.initializers.LeCunNormal()

        
        self.win_size=win_size
        self.query_input=None
        self.batch_size=batch_size
        self.n_out=n_out
        with self.init_scope():
            # self.Zu = self.xp.zeros((batch_size,n_out+16),dtype=np.float32)
            self.last_key = L.Linear(n_out+16, self.att_units_size,initialW=initializer)
            self.query = L.Linear(n_mid_units, self.att_units_size,initialW=initializer)
            self.value = L.Linear(self.att_units_size,n_mid_units,initialW=initializer)
            




    def __call__(self, query, key):


            Z=self.last_key(key)
            Z=F.broadcast_to(Z,query.shape)

            hx=self.query(query,n_batch_axes=2)


            attendW=F.softmax(
                F.tanh(
                    self.value(hx+Z,n_batch_axes=2)
                    )
                ,axis=0)

            context=attendW*query


            context=F.sum(context,axis=0,keepdims=False)*self.win_size

        return context


