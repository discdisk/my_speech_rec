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
from utils import myOptimizers,myAdamOptimizers
from utils import LengthOrderSampler
from evaluater import EEvaluator
import pickle
import h5py


        

def load_data(batch_size):
    # dic=pickle.load(open('/home/chenjh/Desktop/csj/dictionary_131326.pkl','rb'))
    dic=pickle.load(open('/home/chenjh/Desktop/csj/newdic_threshold_1.pkl','rb'))
    data_file=h5py.File('/home/chenjh/Desktop/csj/new_xml_noncore_logf40_meanNorm.hdf5','r')
    X=data_file['voice_data']
    Y=data_file['output_data']
    Z=data_file['output_type']

    train=chainer.datasets.TupleDataset(X,chainer.datasets.TupleDataset(Y,Z))


    test_data_file=h5py.File('/home/chenjh/Desktop/csj/new_xml_core_logf40_meanNorm.hdf5','r')
    testX=test_data_file['voice_data']
    testY=test_data_file['output_data']
    testZ=test_data_file['output_type']

    test=chainer.datasets.TupleDataset(testX,chainer.datasets.TupleDataset(testY,testZ))

    # train_lengthOrder = LengthOrderSampler(train)
    # test_lengthOrder = LengthOrderSampler(test)

    # pickle.dump(train_lengthOrder.length_order,open('train_lengthOrder','wb'))
    # pickle.dump(test_lengthOrder.length_order,open('test_lengthOrder','wb'))

    # train_lengthOrder = LengthOrderSampler(train,length_order=pickle.load(open('train_lengthOrder','rb')))
    # test_lengthOrder = LengthOrderSampler(test,length_order=pickle.load(open('test_lengthOrder','rb')))

    # train_iter = chainer.iterators.MultithreadIterator(train, batch_size,n_threads=6,order_sampler=train_lengthOrder)
    # test_iter = chainer.iterators.MultithreadIterator(test, batch_size,n_threads=6,order_sampler=test_lengthOrder)

    train_iter = chainer.iterators.MultithreadIterator(train, batch_size,shuffle=True,n_threads=6)
    test_iter = chainer.iterators.MultithreadIterator(test, batch_size,shuffle=True,n_threads=6)
    print('test_iter loaded')
    return train_iter,test_iter,dic



def load_model(device,batch_size,loss_fun,out):

    model = RNN(n_lstm_layers=3, n_mid_units=512, n_out=out, win_size=9, batch_size=batch_size, att_units_size=256)
    model = L.Classifier(model, lossfun=loss_fun)
    # chainer.serializers.load_npz('ctc_model_lstm_5000iter',model)
    if device>-1:
        model.to_gpu()
    model.compute_accuracy = False
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

    trainer = training.Trainer(updater, (epochs,'epoch'),out='result')

    
    return trainer




def main():
    gpu    = 0
    epoch  = 5
    b_size = 7
    # chainer.global_config.dtype=np.float16
    train_iter,test_iter,dic=load_data(batch_size=b_size)

    utils=util(gpu,dic['blank'])

    model=load_model(gpu,b_size,utils.ctc_loss,len(dic))
    trainer=setup_trainer(model,train_iter,utils.converter,epoch,gpu)



    eval_model=model.copy()
    eval_rnn = eval_model.predictor
    trainer.extend(EEvaluator(test_iter,eval_model,utils.converter,device=gpu), trigger=(50000, 'iteration'))
    trainer.extend(extensions.LogReport(), trigger=(3000, 'iteration'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss']
    ), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval= 10))
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}'), trigger=(50001, 'iteration'))
    trainer.extend(extensions.PlotReport(['main/loss'], trigger=(1000,'iteration'), file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['validation/main/loss'], trigger=(5000, 'iteration'), file_name='validationloss.png'))
    trainer.extend(extensions.PlotReport(['main/loss','validation/main/loss'], trigger=(50000, 'iteration'), file_name='overall_loss.png'))
    # chainer.serializers.load_npz('/home/chenjh/Desktop/ctc_uniLSTM_att_logf40_512unit_word_output/1024cell_att512with9frame_RMSprop_noncore_data_len_order/snapshot_iter_5000', trainer)
    trainer.run()
    if gpu >= 0:
        model.to_cpu() 

    chainer.serializers.save_npz('ctc_model_unilstm', model)


if __name__ == '__main__':
    main()
    # os.system('shutdown')