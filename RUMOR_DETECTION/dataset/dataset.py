import math
import librosa

import numpy as np
from tqdm import tqdm


class Dataset(object):

    def __init__(self, dataset_path, diot_size=100000, hz=4800):
        self.dataset_path = dataset_path
        self.diot_size = diot_size
        self.hz = hz

    def load_file(self):
        temp_li = []
        with open(self.dataset_path, encoding='utf8') as f:
            line = f.readline()
            while line:
                label, att = line.replace('\n', '').split('\t')
                label = int(label)
                emo_li = [[float(i) for i in emo_att.split(',')] for emo_att in att.split('^')]
                temp_li.append([label, emo_li])
                line = f.readline()
        return temp_li

    def audio_emo_entropy(self, emo_li):
        entropy = 0
        max_index, value = 0, 0
        for index, emo in enumerate(emo_li):
            if emo == 0:
                emo = 0.000001
            if emo > value:
                max_index, value = index, emo
            entropy -= math.log2(emo)
        amp = max_index * value
        return entropy, amp

    def audio_like_segment(self, emo_li, value_li, dot_index, sample_size):
        entropy, amp = self.audio_emo_entropy(emo_li=emo_li)
        for i in range(sample_size):
            index = dot_index * (1 / self.hz)
            temp = amp * math.cos(2 * math.pi * entropy * index)
            dot_index += 1
            value_li.append(temp)
        return dot_index

    def audio_like(self):
        att_li = self.load_file()
        temp_li = []
        for label, item_li in att_li:
            sample_size = self.diot_size // len(item_li)
            current_li, dot_index = [], 0
            for emo_li in item_li:
                dot_index = self.audio_like_segment(emo_li=emo_li, value_li=current_li, sample_size=sample_size,
                                                    dot_index=dot_index)
            while len(current_li) < self.diot_size:
                current_li.append(0.0)
            temp_li.append([label, current_li])
        return temp_li

    def audio_mfcc(self):
        audio_li = self.audio_like()
        label_li, image_li = [], []
        for label, item_li in tqdm(audio_li):
            audio_value = np.asarray(item_li)
            feature_image = librosa.feature.mfcc(y=audio_value, sr=self.hz)
            label_li.append(label)
            image_li.append(feature_image)
        label_li = np.asarray(label_li, dtype=np.int32)
        image_li = np.asarray(image_li, dtype=np.float32)
        return label_li, image_li
