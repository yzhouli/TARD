import json

import numpy as np
import tensorflow as tf

from RUMOR_DETECTION.utils.text_util import TextUtil


class EmoDataset(object):

    def __init__(self, mode, dataset_path, batch_size, word_size, embedding_model, vector_size, china):
        self.batch_size = batch_size
        self.china = china
        self.embedding_model = embedding_model
        self.word_size = word_size
        self.vector_size = vector_size
        self.dataset_path = dataset_path
        if 'train' == mode:
            self.data = json.load(open(f'{dataset_path}/train.json', encoding='utf8'))
        else:
            self.data = json.load(open(f'{dataset_path}/test.json', encoding='utf8'))

    def process_index(self, index):
        index = tf.cast(index, tf.int32)
        return index

    def get_all(self):
        index_li = np.asarray([int(i) for i in self.data])
        data_db = tf.data.Dataset.from_tensor_slices(index_li)
        data_db = data_db.map(self.process_index).shuffle(1000000).batch(self.batch_size)
        return data_db

    def get_item(self, index_li):
        index_li = index_li.numpy()
        wv_matrix, topic_label = [], []
        for index in index_li:
            word_wv, label = self.iteration(index=index)
            wv_matrix.append(word_wv)
            topic_label.append(label)
        wv_matrix = np.asarray(wv_matrix, dtype=np.float32)
        topic_label = np.asarray(topic_label, dtype=np.int32)
        return wv_matrix, topic_label

    def iteration(self, index):
        index = str(index)
        local_json = self.data[index]
        word_wv = TextUtil.normal_words(words=local_json['context'], word_size=self.word_size,
                                        embedding_model=self.embedding_model,
                                        china=self.china)
        label = local_json['label']
        return word_wv, label

    def len(self):
        return len(self.data.keys())
