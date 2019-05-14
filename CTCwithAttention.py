import os
import argparse
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

import chainer
from chainer import training
from chainer.training import extensions

from Modified_module.Modified_Optimizers import myOptimizers, myAdamOptimizers
from Modified_module.Modified_Classifier import MYClassifier
from Modified_module.Modified_Iterator import My_SerialIterator

from models.model_maxpooling import RNN

from tools.load_data import load_data
from tools.utils import util






def load_model(device, batch_size,unit_size,n_layers, loss_fun, out):


    model = RNN(n_lstm_layers=n_layers, n_mid_units=unit_size, n_out=out, win_size=7, batch_size=batch_size, att_units_size=int(unit_size/4), dropout=0.5)
    model = MYClassifier(model, lossfun=loss_fun)
    # chainer.serializers.load_npz('snapshot_iter_30000',model)

    if device > -1:
        model.to_gpu()
    # model.compute_accuracy = False
    chainer.using_config('train', True)
    # chainer.set_debug(True)
    return model


def setup_trainer(model, data_iter, convert, epochs, device, folder_name):
    optimizer = myAdamOptimizers()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(5))
    # optimizer.add_hook(chainer.optimizer_hooks.GradientHardClipping(-1,1))
    # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0000001))

    updater = training.updaters.StandardUpdater(data_iter, optimizer, device=device, converter=convert)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out=folder_name)
    return trainer







def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--e',type=int,default=5)
    parser.add_argument('--b',type=int,default=40)
    parser.add_argument('--u',type=int,default=320)
    parser.add_argument('--n',type=int,default=3)
    args = parser.parse_args()

    epoch     = args.e
    b_size    = args.b
    unit_size = args.u
    n_layers  = args.n
    gpu = 0

    out_folder = f'result_{unit_size}_{n_layers}layer_LSTM_frameatt_2019_0510'

    # chainer.global_config.dtype=np.float16
    train_iter, test_iter, word_dic, char_dic = load_data(batch_size=b_size)

    utils = util(gpu, word_dic['blank'])





    model = load_model(gpu, b_size,unit_size,n_layers, utils.ctc_loss, (len(word_dic), len(char_dic)))
    model.set_utils(utils)

    trainer = setup_trainer(model, train_iter, utils.converter, epoch, gpu, out_folder)

    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    trainer.extend(extensions.Evaluator(test_iter, eval_model, utils.converter, device=gpu), trigger=(10000, 'iteration'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'validation/main/loss']), trigger=(5000, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}'), trigger=(10000, 'iteration'))

    trainer.extend(extensions.PlotReport(                        ['main/loss'], trigger=( 1000, 'iteration'), file_name='loss_main.png'))
    trainer.extend(extensions.PlotReport(             ['validation/main/loss'], trigger=(10000, 'iteration'), file_name='loss_validation.png'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], trigger=(10000, 'iteration'), file_name='loss_overall.png'))

    trainer.extend(extensions.PlotReport(                           ['main/accuracy'], trigger=( 1000, 'iteration'), file_name='wer_main.png'))
    trainer.extend(extensions.PlotReport(                ['validation/main/accuracy'], trigger=(10000, 'iteration'), file_name='wer_validation.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/accuracy'], trigger=(10000, 'iteration'), file_name='wer_overall.png'))


    # chainer.serializers.load_npz(f'/home/chenjh/Desktop/CTCwithAttention_np_2019_04_27/{out_folder}/snapshot_iter_30000', trainer)
    trainer.run()
    if gpu >= 0:
        model.to_cpu()


    chainer.serializers.save_npz('snapshot_iter_', model)


if __name__ == '__main__':
    main()
    # os.system('shutdown')
