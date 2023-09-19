import json

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from RUMOR_DETECTION.utils.text_util import TextUtil


class EmoUtil(object):

    def __init__(self, emo_model, embedding_model, word_size, china):
        self.emo_model = emo_model
        self.embedding_model = embedding_model
        self.word_size = word_size
        self.china = china
        self.softmax = tf.keras.layers.Softmax()

    def load_file(self, path):
        temp_li = []
        with open(path, encoding='utf8') as f:
            line = f.readline()
            while line:
                att_li = line.replace('\n', '').split('\t')
                comment_time, context = int(att_li[2]), att_li[6]
                temp_li.append([comment_time, context])
                line = f.readline()
        temp_li.sort(key=lambda x: x[0])
        return [i[1] for i in temp_li]

    def wv_file(self, context_li):
        wv_li = []
        for context in context_li:
            wv_context = TextUtil.normal_words(words=context, word_size=self.word_size,
                                               embedding_model=self.embedding_model,
                                               china=self.china)
            wv_li.append(wv_context)
        return wv_li

    def normal_predict(self, predict_li):
        file_line = ''
        for index, predict_line in enumerate(predict_li.numpy()):
            line = f'{predict_line[0]},{predict_line[1]},{predict_line[2]}'
            if index == 0:
                file_line += line
            else:
                file_line += f'^{line}'
        return file_line

    def emo_file(self, file_path):
        context_li = self.load_file(path=file_path)
        wv_li = self.wv_file(context_li=context_li)
        wv_matrix = tf.cast(np.asarray(wv_li, dtype=np.float32), dtype=tf.float32)
        predict_li = self.emo_model(wv_matrix)
        predict_li = self.softmax(predict_li)
        file_line = self.normal_predict(predict_li)
        return file_line

    def data_preprocess(self, dataset_path, save_path, train=True):
        json_path = f'{dataset_path}/test.json'
        db_path = f'{save_path}/test.txt'
        if train:
            json_path = f'{dataset_path}/train.json'
            db_path = f'{save_path}/train.txt'
        file_li = json.load(open(json_path))
        with open(db_path, 'w+', encoding='utf8') as f:
            for index in tqdm(file_li, desc=f'train: {train}'):
                label = file_li[index]['label']
                file_path = file_li[index]['path']
                emo_line = self.emo_file(file_path=f'{dataset_path}/{file_path}')
                f.write(f'{label}\t{emo_line}\n')
