import os
import tensorflow as tf
from BAUIDU_EMO.model.lstm import LSTM
from BAUIDU_EMO.train.train import EmoTrain
from RUMOR_DETECTION.model.model import CNN
from RUMOR_DETECTION.train.Train import Train
from RUMOR_DETECTION.utils.emo_util import EmoUtil
from RUMOR_DETECTION.utils.text_util import TextUtil


def load_wv_model(path, dataset_path, vector_size, select_id):
    dictionary_path = f'{path}/dictionary.txt'
    wv_path = f'{path}/wv-model'
    if os.path.exists(wv_path):
        return TextUtil.load(wv_path)
    else:
        return TextUtil.build_word_vector(dictionary_path=dictionary_path, dir_path=dataset_path,
                                          vector_size=vector_size,
                                          select_id=select_id, save_path=wv_path)


def train_emo_model(emo_dataset_path, embedding_model, save_path, evl_model, train_model, epochs, is_train):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.66)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    # 查看日志信息若包含gpu信息，就是使用了gpu
    word_size = 29
    vector_size = 100
    out_class = 3
    learning_rate = 0.001
    batch_size = 512
    china = True
    temp_path = 'RUMOR_DETECTION/data/temp/emo-model'
    train = EmoTrain(learning_rate=learning_rate, model=train_model, dataset_path=emo_dataset_path,
                     batch_size=batch_size,
                     epochs=epochs,
                     vector_size=vector_size, word_size=word_size, save_path=save_path, embedding_model=embedding_model,
                     china=china, depth=out_class, evl_model=evl_model, is_train=is_train, temp_path=temp_path)
    train.train()


def data_preprocess(dataset_path, emo_dataset_path, wv_model_path, emo_path, db_path):
    h_dim = 64
    word_size = 29
    vector_size = 100
    out_class = 3
    select_id = 6
    embedding_model = load_wv_model(path=wv_model_path, dataset_path=dataset_path, vector_size=vector_size,
                                    select_id=select_id)
    evl_model = LSTM(h_dim=h_dim, word_size=word_size, vector_size=vector_size, out_class=out_class,
                     train=False)
    train_model = LSTM(h_dim=h_dim, word_size=word_size, vector_size=vector_size, out_class=out_class,
                       train=False)
    epochs, is_train = 2, False
    if not os.path.exists(f'{emo_path}.index'):
        epochs, is_train = 30, True
    train_emo_model(emo_dataset_path=emo_dataset_path, embedding_model=embedding_model, evl_model=evl_model,
                    save_path=emo_path, train_model=train_model, epochs=epochs, is_train=is_train)

    eu = EmoUtil(emo_model=evl_model, embedding_model=embedding_model, word_size=word_size, china=True)
    eu.data_preprocess(dataset_path=dataset_path, save_path=db_path, train=True)
    eu.data_preprocess(dataset_path=dataset_path, save_path=db_path, train=False)


def rumor_detection(dataset_path):
    height_size = 20
    weight_size = 196
    out_class = 2
    hz = 4800
    dot_size = 100000
    learning_rate = 0.003
    batch_size = 64
    epochs = 50
    temp_path = 'RUMOR_DETECTION/data/temp/rumor-model'
    model = CNN(height_size=height_size, weight_size=weight_size, out_class=out_class, train=True)
    evl_model = CNN(height_size=height_size, weight_size=weight_size, out_class=out_class, train=False)

    train = Train(model=model, dataset_path=dataset_path, hz=hz, dot_size=dot_size, learning_rate=learning_rate,
                  batch_size=batch_size, depth=out_class, epochs=epochs, evl_model=evl_model, temp_path=temp_path)
    train.train()


if __name__ == '__main__':
    wv_model_path = 'RUMOR_DETECTION/data/wv/'
    dataset_path = 'E:/IPM/weibo'
    emo_dataset_path = 'E:/IPM/emo'
    db_path = 'RUMOR_DETECTION/data/database/'
    emo_path = 'RUMOR_DETECTION/data/emo/emo-model'
    # data_preprocess(dataset_path=dataset_path, emo_dataset_path=emo_dataset_path, wv_model_path=wv_model_path,
    #                 emo_path=emo_path, db_path=db_path)
    rumor_detection(dataset_path=db_path)
