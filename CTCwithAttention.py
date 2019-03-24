import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer import reporter as reporter_module
from chainer import function
import numpy as np

from model import RNN
from utils import util

from Modified_Optimizers import myOptimizers,myAdamOptimizers
from Modified_Classifier import MYClassifier
from Modified_Evaluater import Evaluator
from Modified_Iterator import My_SerialIterator

import pickle

        

def load_data(batch_size):
    word_dic=pickle.load(open('/home/chenjh/Desktop/csj/npData_word_dic.pkl','rb'))
    char_dic=pickle.load(open('/home/chenjh/Desktop/csj/npData_char_dic.pkl','rb'))

    # Data structure
    # list of dic
    #########################
    # {'ori_sound':file_name,
    # 'fbank_feat':file_name,
    # 'frame_length':fbank_feat.shape[0],
    # 'text_length':len(text),
    # 'PlainOrthographicTranscription':text,
    # 'PhoneticTranscription':PhoneticTranscription,
    # 'output_word':word target,
    # 'output_char':char target}

    # data_file=np.load('/home/chenjh/Desktop/csj/new_xml_noncore_logf40_meanNorm.npy')
    # path='/home/chenjh/Desktop/csj/new_xml_noncore_logf40_meanNorm/fbank_feat/'
    # X=[path+data['fbank_feat'] for data in data_file]
    # Y=[data['output_word'] for data in data_file]
    # Z=[data['output_char'] for data in data_file]

    # data_len=np.array([data['text_length'] for data in data_file],dtype=np.int32)

    # train=chainer.datasets.TupleDataset(X,chainer.datasets.TupleDataset(Y,Z))



    test_data_file=np.load('/home/chenjh/Desktop/csj/new_xml_core_logf40_meanNorm.npy')
    path='/home/chenjh/Desktop/csj/new_xml_core_logf40_meanNorm/fbank_feat/'
    testX=[path+test_data['fbank_feat'] for test_data in test_data_file]
    testY=[test_data['output_word'] for test_data in test_data_file]
    testZ=[test_data['output_char'] for test_data in test_data_file]

    test=chainer.datasets.TupleDataset(testX,chainer.datasets.TupleDataset(testY,testZ))

    train=test[5000:]
    test=test[:5000]




    # train_iter = chainer.iterators.MultithreadIterator(train, batch_size,shuffle=True,n_threads=6)
    # test_iter = chainer.iterators.MultithreadIterator(test, batch_size,shuffle=True,n_threads=6)
    train_iter = My_SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size)
    print('test_iter loaded')
    return train_iter,test_iter,word_dic,char_dic



def load_model(device,batch_size,loss_fun,out):

    model = RNN(n_lstm_layers=1, n_mid_units=512, n_out=out, win_size=9, batch_size=batch_size, att_units_size=256)
    def loss(x,t):
        losss=F.softmax_cross_entropy(x,t)
        print(losss)
        return losss
    model = L.Classifier(model, lossfun=loss)
    # chainer.serializers.load_npz('ctc_model_lstm_5000iter',model)
    if device>-1:
        model.to_gpu()
    # model.compute_accuracy = False
    chainer.using_config('train', True)
    # chainer.set_debug(True)
    return model



def setup_trainer(model,data_iter,convert,epochs,device):
    optimizer = myAdamOptimizers()
    # optimizer = myOptimizers(lr=0.00001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(5))
    # optimizer.add_hook(chainer.optimizer_hooks.GradientHardClipping(-1,1))
    # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0000001))


    updater = training.updaters.StandardUpdater(data_iter, optimizer,
        device=device ,converter=convert)

    # trainer = training.Trainer(updater, (epochs,'epoch'),out='all_data_1024cell_att512with9frame_Adam_hardClip')
    trainer = training.Trainer(updater, (epochs,'epoch'),out='result512_3layer_withoutmark_frameatt')

    
    return trainer




def main():
    gpu    = 0
    epoch  = 5
    b_size = 7
    # chainer.global_config.dtype=np.float16
    train_iter,test_iter,word_dic,char_dic=load_data(batch_size=b_size)


    utils=util(gpu,word_dic['blank'])

    model=load_model(gpu,b_size,utils.ctc_loss,len(word_dic))
    trainer=setup_trainer(model,train_iter,utils.converter,epoch,gpu)



    eval_model=model.copy()
    eval_rnn = eval_model.predictor
    trainer.extend(Evaluator(test_iter,eval_model,utils.converter,device=gpu), trigger=(10000, 'iteration'))
    trainer.extend(extensions.LogReport(), trigger=(3000, 'iteration'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss']
    ), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval= 10))
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}'), trigger=(10000, 'iteration'))
    trainer.extend(extensions.PlotReport(['main/loss'], trigger=(1000,'iteration'), file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy'], trigger=(1000,'iteration'), file_name='accuracy.png'))
    # trainer.extend(extensions.PlotReport(['main/word_loss'], trigger=(1000,'iteration'), file_name='word_loss.png'))
    # trainer.extend(extensions.PlotReport(['main/char_loss'], trigger=(1000,'iteration'), file_name='char_loss.png'))
    trainer.extend(extensions.PlotReport(['validation/main/loss'], trigger=(10000, 'iteration'), file_name='validationloss.png'))
    trainer.extend(extensions.PlotReport(['validation/main/accuracy'], trigger=(10000,'iteration'), file_name='validationaccuracy.png'))
    # trainer.extend(extensions.PlotReport(['validation/main/word_loss'], trigger=(5000, 'iteration'), file_name='validationword_loss.png'))
    # trainer.extend(extensions.PlotReport(['validation/main/char_loss'], trigger=(5000, 'iteration'), file_name='validationchar_loss.png'))
    trainer.extend(extensions.PlotReport(['main/loss','validation/main/loss'], trigger=(10000, 'iteration'), file_name='overall_loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/accuracy'], trigger=(10000, 'iteration'), file_name='overall_accuracy.png'))
    # trainer.extend(extensions.PlotReport(['main/word_loss','validation/main/word_loss'], trigger=(50000, 'iteration'), file_name='overall_word_loss.png'))
    # trainer.extend(extensions.PlotReport(['main/char_loss','validation/main/char_loss'], trigger=(50000, 'iteration'), file_name='overall_char_loss.png'))
    # chainer.serializers.load_npz('/home/chenjh/Desktop/CTC_uniLSTM_att_log40_npData/result512_3layer_withoutmark_frameatt/snapshot_iter_37037', trainer)
    trainer.run()
    if gpu >= 0:
        model.to_cpu() 

    chainer.serializers.save_npz('ctc_model_lstm_125125iter', model)


if __name__ == '__main__':
    main()
    # os.system('shutdown')