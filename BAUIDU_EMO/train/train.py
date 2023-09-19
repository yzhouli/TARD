import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, models
from tqdm import tqdm

from BAUIDU_EMO.dataset.dataset import EmoDataset
from RUMOR_DETECTION.utils.model_util import ModelUtil


class EmoTrain(object):

    def __init__(self, learning_rate, model, dataset_path, batch_size, epochs, vector_size, word_size, save_path,
                 embedding_model, china, depth, evl_model, is_train, temp_path):
        self.learning_rate = learning_rate
        self.model = model
        self.temp_path = temp_path
        self.china = china
        self.depth = depth
        self.epochs = epochs
        self.is_train = is_train
        self.evl_model = evl_model
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.vector_size = vector_size
        self.word_size = word_size
        self.save_path = save_path
        self.embedding_model = embedding_model
        self.train_db, self.test_db = None, None

    def data_init(self):

        self.train_db = EmoDataset(mode='train', dataset_path=self.dataset_path,
                                   batch_size=self.batch_size,
                                   word_size=self.word_size, embedding_model=self.embedding_model,
                                   vector_size=self.vector_size, china=self.china)

        self.test_db = EmoDataset(mode='test', dataset_path=self.dataset_path,
                                  batch_size=self.batch_size,
                                  word_size=self.word_size, embedding_model=self.embedding_model,
                                  vector_size=self.vector_size, china=self.china)

    def iteration(self, name, epoch):
        data_train = self.train_db
        current_model = self.model
        if 'test' == name:
            data_train = self.test_db
            current_model = self.evl_model
        optimizer = optimizers.Adam(self.learning_rate)
        acc_all, loss_all = 0, 0
        if 'train' == name:
            colour = '#FF0000'
        else:
            colour = '#00FF00'
        pbar = tqdm(total=data_train.len() // self.batch_size + 1, colour=colour)
        with tf.device('/gpu:0'):
            for index, item in enumerate(data_train.get_all()):
                wv_matrix, topic_label = data_train.get_item(index_li=item)
                total, true_total, loss_total = 0, 0, 0
                with tf.GradientTape() as tape:
                    label_matrix = tf.one_hot(topic_label, depth=self.depth)
                    out_matrix = current_model(wv_matrix)
                    current_true_total = ModelUtil.acc(label_matrix=topic_label, out_matrix=out_matrix)
                    total, true_total = total + len(topic_label), true_total + current_true_total

                    loss = tf.losses.categorical_crossentropy(label_matrix, out_matrix, from_logits=True)
                    loss = tf.reduce_mean(loss)
                    loss_total += float(loss) * len(topic_label)

                    acc_all, loss_all = round(true_total / total, 4), round(loss_total / total, 4)
                    pbar.desc = f'epoch: {epoch}, name: {name}, loss: {loss_all}, accuracy: {acc_all}'
                    pbar.update(1)

                    if 'train' == name:
                        grads = tape.gradient(loss, current_model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, current_model.trainable_variables))

        return acc_all

    def train(self):
        self.data_init()
        max_acc = -1
        for epoch in range(self.epochs):
            if self.is_train:
                self.iteration(name='train', epoch=epoch)
                ModelUtil.model_save(save_path=self.temp_path, model=self.model)
                ModelUtil.model_load(save_path=self.temp_path, model=self.evl_model)
            acc = self.iteration(name='test', epoch=epoch)
            if not self.is_train:
                ModelUtil.model_load(save_path=self.save_path, model=self.evl_model)
            if acc > max_acc and self.is_train:
                max_acc = acc
                ModelUtil.model_save(save_path=self.save_path, model=self.evl_model)
        print(f'max accuracy: {max_acc}')
