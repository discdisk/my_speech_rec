import numpy as np
import pickle

def load_data(batch_size):
    dataset=''
    word_dic = pickle.load(open('/home/chenjh/Desktop/csj/making_correct_data/APS_npData_word_dic.pkl', 'rb'))
    char_dic = pickle.load(open('/home/chenjh/Desktop/csj/making_correct_data/APS_npData_char_dic.pkl', 'rb'))

    # Data structure
    # list of dic
    #########################
    # {'ori_filename': file,
    # 'core_noncore': data_folder[:-1],
    # 'ori_sound':f'ori_sound_{save_count}{count}.npy',
    # 'fbank_feat':f'fbank_feat_{save_count}{count}.npy',
    # 'fbank_feat_mean_norm':f'fbank_feat_mean_norm_{save_count}{count}.npy',
    # 'frame_length':fbank_feat.shape[0],
    # 'PlainOrthographicTranscription':text,
    # 'PhoneticTranscription':PhoneticTranscription}

    data_file = np.load('/home/chenjh/Desktop/csj/making_correct_data/new_xml_logf40_meanNorm_APS.npy')
    path = '/home/chenjh/Desktop/csj/making_correct_data/new_xml_logf40_meanNorm_APS/fbank_feat_mean_norm/'
    X = [path + data['fbank_feat_mean_norm'] for data in data_file]
    Y = [data['output_word'] for data in data_file]
    Z = [data['output_char'] for data in data_file]


    train = chainer.datasets.TupleDataset(X, chainer.datasets.TupleDataset(Y, Z))

    test_data_file = np.load('/home/chenjh/Desktop/csj/making_correct_data/new_xml_logf40_meanNorm_APS_test.npy')
    path = '/home/chenjh/Desktop/csj/making_correct_data/new_xml_logf40_meanNorm_APS_test/fbank_feat_mean_norm/'
    testX = [path + test_data['fbank_feat_mean_norm'] for test_data in test_data_file]
    testY = [test_data['output_word'] for test_data in test_data_file]
    testZ = [test_data['output_char'] for test_data in test_data_file]

    test = chainer.datasets.TupleDataset(testX, chainer.datasets.TupleDataset(testY, testZ))

    # train_iter = chainer.iterators.MultithreadIterator(train, batch_size,shuffle=True,n_threads=6)
    # test_iter = chainer.iterators.MultithreadIterator(test, batch_size,shuffle=True,n_threads=6)
    train_iter = My_SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test[:len(test)//batch_size*batch_size], batch_size, shuffle=False, repeat=False)
    print('test_iter loaded')
    return train_iter, test_iter, word_dic, char_dic