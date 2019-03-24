
import chainer.functions as F
import chainer.links as L
import chainer
import numpy as np
from chainer.backends import cuda




class RNN(chainer.Chain):

    def __init__(self, n_lstm_layers, n_mid_units, n_out, win_size, batch_size, att_units_size, dropout=0.5 ):
        super(RNN, self).__init__()

        initializer = chainer.initializers.Normal()
        self.init_last_shape=(batch_size,n_mid_units+16)
        self.win_size=win_size
        self.pad_size=1
        self.batch_size=batch_size
        self.n_mid_units=n_mid_units
        xp=cuda.cupy

        self.pad_zero=xp.zeros((batch_size,self.pad_size),dtype=self.xp.int32)
        with self.init_scope():
            # self.layer_norm1=L.LayerNormalization()
            self.audio_fc = L.Linear(200, n_mid_units,initialW=initializer)
            # self.batch_norm1=L.BatchNormalization(n_mid_units)

            self.audio_encoder = L.NStepLSTM(n_lstm_layers,
                                    in_size=n_mid_units, \
                                    out_size=n_mid_units,
                                    dropout=dropout,)


            self.target_embed = L.EmbedID(n_out, n_mid_units)
            self.target_encoder = L.NStepLSTM(n_lstm_layers,
                                        in_size=n_mid_units, \
                                        out_size=n_mid_units,
                                        dropout=dropout,)

            self.decoder = L.NStepLSTM(n_lstm_layers,
                                        in_size=n_mid_units, \
                                        out_size=n_mid_units,
                                        dropout=dropout,)



            # self.layer_norm2=L.LayerNormalization()
            # self.batch_norm2=L.BatchNormalization(n_mid_units)





            self.attend = Attention(n_mid_units, n_out, win_size, batch_size, att_units_size)

            self.word_output = L.Linear(n_mid_units, n_out,initialW=initializer)


    def __call__(self, data):
        xs, word_label=data
        out_len=[len(w[0]) for w in word_label]
        # forward calculation
        result=[]
        # xs=[F.concat((F.concat((self.pad_zero,gt),axis=0),self.pad_zero),axis=0) for gt in xs]
        xs=F.pad_sequence(xs)

        h=F.relu(self.audio_fc(xs,n_batch_axes=2))

        _, _, h=self.audio_encoder(None, None, list(h))
        h=F.stack(h,axis=0)


        word_label=F.pad_sequence([w[0] for w in word_label])
        word_label=F.concat((self.pad_zero,word_label),axis=1)


        word_embed=self.target_embed(word_label).reshape(self.batch_size,-1,self.n_mid_units)
        _, _, word_embed=self.target_encoder(None, None, list(word_embed))

        word_embed=F.stack(word_embed,axis=1)

        for key in word_embed:
            context=self.attend(h,key)
            _, _, out=self.decoder(None, None, list(context))
            out=self.word_output(F.stack(out,axis=0)[:,-1,:].reshape(self.batch_size,self.n_mid_units))
            result.append(out)




        result=F.stack(result,axis=1)
        result=F.concat([r[:l] for r,l in zip(result,out_len)],axis=0)

        
        return result







class Attention(chainer.Chain):
    """docstring for Attention"""
    def __init__(self, n_mid_units, n_out, win_size, batch_size, att_units_size, device=-1):
        super(Attention,self).__init__()

        self.win_size=win_size
        self.batch_size=batch_size
        self.att_units_size=att_units_size
        self.n_mid_units=n_mid_units
        self.n_out=n_out
        xp=cuda.cupy

        initializer = chainer.initializers.LeCunNormal()

        with self.init_scope():
            self.key_layer = L.Linear(n_mid_units, self.att_units_size,initialW=initializer)
            self.query_layer = L.Linear(n_mid_units, self.att_units_size,initialW=initializer)
            self.att_cal = L.Linear(self.att_units_size,1,initialW=initializer)

    def __call__(self, input_query,key):
            
        hx=self.query_layer(input_query,n_batch_axes=2)

        Z=self.key_layer(key)
        Z=F.broadcast_to(F.expand_dims(Z,axis=1),(hx.shape))

        attend=hx+Z

        attend=F.sigmoid(self.att_cal(
                        F.tanh(attend.reshape(-1,self.att_units_size))
                        ).reshape(self.batch_size,-1,1))




        context=input_query*attend



        return context
