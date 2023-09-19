import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf


class ModelUtil(object):
    softmax = tf.keras.layers.Softmax()

    @staticmethod
    def max_index(pred_li):
        index = 0
        temp = -1000
        for i, num in enumerate(pred_li):
            if num > temp:
                temp = num
                index = i
        return index

    @staticmethod
    def normal(predict_li, depth):
        result_li = []
        for att_li in predict_li.numpy():
            index = ModelUtil.max_index(att_li)
            result_li.append(index)
        result_li = np.asarray(result_li, dtype=np.int32)
        result_li = tf.cast(result_li, dtype=tf.int32)
        result_li = tf.one_hot(result_li, depth=depth)
        return result_li

    @staticmethod
    def evaluation_show(precision_li, recall_li, f1_score_li):
        show_li = ['' for i in range(len(precision_li))]
        for i in range(len(precision_li)):
            show_li[i] += f'prec: {precision_li[i]}, recall: {recall_li[i]}, F1: {f1_score_li[i]}\n'
        return show_li

    @staticmethod
    def acc(label_matrix, out_matrix):
        out_matrix = out_matrix.numpy()
        true_total = 0
        for i in range(len(label_matrix)):
            if label_matrix[i] == -1:
                continue
            pred = tf.nn.softmax(out_matrix[i])
            pred_index = ModelUtil.max_index(pred_li=pred)
            if pred_index == label_matrix[i]:
                true_total += 1
        return true_total

    @staticmethod
    def evaluation(y_test, y_predict, depth, is_show: bool, softmax=softmax):
        y_predict = softmax(y_predict)
        y_predict = ModelUtil.normal(y_predict, depth=depth)
        metrics = classification_report(y_test, y_predict, output_dict=True)
        precision_li, recall_li, f1_score_li = [], [], []
        for i in range(depth):
            precision_li.append(metrics[f'{i}']['precision'])
            recall_li.append(metrics[f'{i}']['recall'])
            f1_score_li.append(metrics[f'{i}']['f1-score'])
        if is_show:
            return ModelUtil.evaluation_show(precision_li, recall_li, f1_score_li)
        return precision_li, recall_li, f1_score_li

    @staticmethod
    def model_save(save_path, model):
        model.save_weights(filepath=save_path)

    @staticmethod
    def model_load(save_path, model):
        return model.load_weights(save_path)
