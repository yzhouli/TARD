from tqdm import tqdm

from RUMOR_DETECTION.dataset.dataset import Dataset
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, models

from RUMOR_DETECTION.utils.model_util import ModelUtil


class Train(object):

    def __init__(self, model, dataset_path, hz, dot_size, learning_rate, batch_size, depth, epochs, evl_model,
                 temp_path):
        self.epochs = epochs
        self.evl_model = evl_model
        self.dataset_path = dataset_path
        self.hz = hz
        self.temp_path = temp_path
        self.model = model
        self.depth = depth
        self.dot_size = dot_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_db, self.test_db = None, None

    def data_init(self):
        data_db = Dataset(dataset_path=f'{self.dataset_path}/train.txt', diot_size=self.dot_size, hz=self.hz)
        self.train_db = data_db.audio_mfcc()
        data_db = Dataset(dataset_path=f'{self.dataset_path}/test.txt', diot_size=self.dot_size, hz=self.hz)
        self.test_db = data_db.audio_mfcc()

    def process_index(self, label, image):
        label = tf.cast(label, tf.int32)
        image = tf.cast(image, tf.float32)
        return label, image

    def iteration(self, epoch, name):
        label_li, image_li = self.test_db
        current_model = self.evl_model
        if 'train' == name:
            label_li, image_li = self.train_db
            current_model = self.model
        optimizer = optimizers.Adam(self.learning_rate)
        data_db = tf.data.Dataset.from_tensor_slices((label_li, image_li))
        data_db = data_db.map(self.process_index).shuffle(100000).batch(self.batch_size)
        acc_all, loss_all = 0, 0
        pbar = tqdm(total=len(label_li) // self.batch_size + 1)
        for (label_li, image_li) in data_db:
            total, true_total, loss_total = 0, 0, 0
            with tf.GradientTape() as tape:
                label_matrix = tf.one_hot(label_li, depth=self.depth)
                out_matrix = current_model(image_li)
                current_true_total = ModelUtil.acc(label_matrix=label_li, out_matrix=out_matrix)
                total, true_total = total + len(label_li), true_total + current_true_total

                loss = tf.losses.categorical_crossentropy(label_matrix, out_matrix, from_logits=True)
                loss = tf.reduce_mean(loss)
                loss_total += float(loss) * len(label_li)

                acc_all, loss_all = round(true_total / total, 4), round(loss_total / total, 4)
                pbar.desc = f'epoch: {epoch}, name: {name}, loss: {loss_all}, accuracy: {acc_all}'
                pbar.update(1)

                if 'train' == name:
                    grads = tape.gradient(loss, current_model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, current_model.trainable_variables))
        return acc_all

    def train(self):
        print('database init')
        self.data_init()
        max_acc = -1
        print('model training')
        for epoch in range(self.epochs):
            self.iteration(name='train', epoch=epoch)
            ModelUtil.model_save(save_path=self.temp_path, model=self.model)
            ModelUtil.model_load(save_path=self.temp_path, model=self.evl_model)
            acc = self.iteration(name='test', epoch=epoch)
            max_acc = max(acc, max_acc)
        print(f'max accuracy: {max_acc}')
