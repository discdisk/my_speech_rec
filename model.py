
import chainer.functions as F
import chainer.links as L
import chainer
import numpy as np
from chainer.backends import cuda

def argsort_list_descent(lst):
    return numpy.argsort([-len(x.data) for x in lst]).astype('i')


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

        initializer = chainer.initializers.Normal()
        
        with self.init_scope():
            # self.layer_norm1=L.LayerNormalization()
            self.fc1 = L.Linear(None, n_mid_units,initialW=initializer)
            # self.batch_norm1=L.BatchNormalization(n_mid_units)

            self.lstm = L.NStepLSTM(n_lstm_layers,
                                    in_size=n_mid_units, \
                                    out_size=n_mid_units,
                                    dropout=dropout,)
            # self.layer_norm2=L.LayerNormalization()
            # self.batch_norm2=L.BatchNormalization(n_mid_units)





            self.attend = Attention(n_mid_units, n_out, win_size, batch_size, att_units_size)



    def __call__(self, xs):
    # forward calculation

    # sort input sequence by length
        indices = argsort_list_descent(xs)
        indices_array = xp.array(indices)

        xs = permutate_list(xs, indices, inv=False)

    # from shape(B,S,V)==>(S,B,V)
        trans_x = transpose_sequence.transpose_sequence(xs)
        hy, cy = None, None
        for x in xs:

            h1 = F.relu(self.fc1(x))
            hy, cy, ys = self.l2(hy, cy, h1)

            result=self.attend(ys)




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
            self.Zu = self.xp.zeros((batch_size,n_out+16),dtype=np.float32)
            self.last_key = L.Linear(n_out+16, self.att_units_size,initialW=initializer)
            self.query = L.Linear(n_mid_units, self.att_units_size,initialW=initializer)
            self.value = L.Linear(self.att_units_size,n_mid_units,initialW=initializer)
            self.word_output = L.Linear(n_mid_units, n_out,initialW=initializer)
            self.type_output = L.Linear(n_mid_units, 16,initialW=initializer)


    def reset(self):
        self.query_input=None
        self.Zu=self.Zu = self.xp.zeros((self.batch_size,self.n_out+16),dtype=np.float32)


    def __call__(self, x):

        if self.query_input is None:
            self.query_input=self.query(x)[None]

        shape=self.query_input.shape
        if shape[1]>x.shape[0]:
            self.query_input=self.query_input[:,:x.shape[0],:]
            self.Zu+self.Zu[:,:x.shape[0],:]

        self.query_input=F.concat(self.query_input,self.query(x)[None])
        if shape[0] == self.win_size:

            Z=self.last_key(self.Zu)
            Z=F.broadcast_to(Z,self.query_input.shape)


            attendW=F.softmax(
                F.tanh(
                    self.value(hx+Z,n_batch_axes=2)
                    )
                ,axis=0)

            context=attendW*self.query_input


            context=F.sum(context,axis=0,keepdims=False)*self.win_size
            result=self.output(context)
            result_type=self.type_output(context)

            self.Zu=F.concat([result,result_type]).data
            self.query_input=self.query_input[1:]

        return result,result_type


