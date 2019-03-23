
import chainer.functions as F
import chainer.links as L
import chainer
import numpy as np
from chainer.backends import cuda




class RNN(chainer.Chain):

    def __init__(self, n_lstm_layers, n_mid_units, n_out, win_size, batch_size, att_units_size, dropout=0.5 ):
        super(RNN, self).__init__()

        initializer = chainer.initializers.Normal()
        n_word_out=n_out[0]
        n_char_out=n_out[1]
        with self.init_scope():
            self.embed_word_out=L.EmbedID(n_word_out,n_mid_units)
            self.fc1 = L.Linear(None, n_mid_units,initialW=initializer)
            self.char_out = L.Linear(n_mid_units, n_char_out,initialW=initializer)
            self.shared_encoder = L.NStepLSTM(1,
                                    in_size=n_mid_units, \
                                    out_size=n_mid_units,
                                    dropout=dropout,)
            self.frame_encoder = L.NStepLSTM(n_lstm_layers,
                                    in_size=n_mid_units, \
                                    out_size=n_mid_units,
                                    dropout=dropout,)
            self.output_encoder = L.NStepLSTM(n_lstm_layers,
                                    in_size=n_mid_units, \
                                    out_size=n_mid_units,
                                    dropout=dropout,)
            self.decoder = L.NStepLSTM(n_lstm_layers,
                                    in_size=n_mid_units, \
                                    out_size=n_mid_units,
                                    dropout=dropout,)
            self.word_out = L.Linear(n_mid_units, n_word_out,initialW=initializer)
            # self.layer_norm2=L.LayerNormalization()
            # self.batch_norm2=L.BatchNormalization(n_mid_units)





            self.attend = Attention(n_mid_units, n_word_out, win_size, batch_size, att_units_size)



    def __call__(self, x,y):
        # forward calculation
        # concat input to one row
        x=F.stack(x,axis=0)
        h=self.fc1(x)
        # split it back
        h=split_axis(h)



        # get low level feature
        _, _, low_feature = self.shared_encoder(None, None, h)

        _, _, frame_encoder = self.frame_encoder(None, None, low_feature)


        y=F.stack(y,axis=0)
        embed_word=self.embed_word_out(y)
        # split it back
        embed_word=split_axis(embed_word)

        # get embed_word feature
        _, _, embed_word_feature = self.output_encoder(None, None, embed_word)

        result=[]
        for output_step in embed_word_feature:
            # do attention between previous output and whole frame
            att_weight=self.attend(output_step,frame_encoder)
            feature=frame_encoder*att_weight
            word_out=self.decoder(feature)
            word_out=self.word_out(word_out[-1])
            result.append(word_out)
        return result









