import os

import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from tqdm import tqdm


class TextUtil(object):
    stop_words = [
        ['am', 'is', 'until', "isn't", 'the', "mustn't", 'themselves', 'wouldn', 'me', 'has', 'won', "doesn't",
         "haven't", 'now', 'mightn', 'yours', 'if', 'any', 'before', 'down', 'that', 'what', 'few', "mightn't",
         "shouldn't", 'couldn', "it's", 'them', "you'll", 'shan', 'during', "aren't", 'than', 've', 'and', 'off',
         'doesn', 'at', 'so', 'can', "hadn't", 'further', 'i', 'he', 'just', 'have', 'd', 'm', 'ourselves', 'whom',
         'on', 'here', 'same', 'weren', 'been', 'when', 'a', 'which', "don't", 'was', "hasn't", 'how', 'these',
         'because', 'then', 'y', "wouldn't", 'ain', 'did', 'do', 'himself', 'are', 'too', 'into', 'over', 'own',
         "you're", 'for', 'most', 'my', 'not', 'while', 'isn', 'our', 'don', 'there', 're', 'herself', 'again',
         'against', 'as', 'after', "should've", "you've", 'only', 'didn', "weren't", 'aren', "she's", 'why', 'very',
         'who', 'she', 'all', 'below', 'hasn', 's', 'hadn', 'him', 't', 'o', 'once', 'those', 'more', 'itself', 'does',
         'each', 'or', 'wasn', 'both', "needn't", 'with', 'about', 'through', 'they', 'll', 'having', 'its', "didn't",
         'nor', 'up', 'an', 'yourself', "wasn't", 'should', 'be', 'haven', 'we', 'needn', 'ma', "shan't", 'myself',
         'being', 'between', 'hers', 'such', 'mustn', 'had', 'of', 'his', 'shouldn', 'some', "that'll", 'no', 'above',
         'will', 'theirs', 'by', 'their', 'from', 'her', 'to', 'doing', 'it', 'this', "won't", 'under', "you'd",
         'where', "couldn't", 'yourselves', 'in', 'out', 'other', 'your', 'you', 'were', 'but', 'ours']]

    @staticmethod
    def chinese_cut(line: str):
        word_li = jieba.cut(line)
        word_line = ' '.join(word_li)
        return word_line

    @staticmethod
    def english_cut(line: str, stop_words=stop_words):
        result_line = ''
        index = 0
        for word in line.split(' '):
            if word in stop_words:
                continue
            if index == 0:
                index = 1
                result_line += word
            else:
                result_line += f' {word}'
        return result_line

    @staticmethod
    def line_cut(line: str, china: bool):
        if china:
            word_line = TextUtil.chinese_cut(line=line)
        else:
            word_line = TextUtil.english_cut(line=line)
        return word_line

    @staticmethod
    def write_dictionary_item(dictionary_path, file_path, select_id, first_write=False, china=True, padding='PADDING'):
        if first_write:
            write_model = 'w+'
        else:
            write_model = 'a+'
        with open(dictionary_path, write_model, encoding='utf-8') as f:
            if first_write:
                f.write(f'{padding}\n')
            with open(file_path, encoding='utf-8') as fr:
                line = fr.readline()
                while line:
                    line = line.replace('\n', '').split('\t')[select_id]
                    word_line = TextUtil.line_cut(line=line, china=china)
                    f.write(f'{word_line}\n')
                    line = fr.readline()

    @staticmethod
    def write_dictionary(dictionary_path, dir_path: str, select_id, index=1, add=True):
        dir_name = dir_path.split('/')[-1]
        for file_name in tqdm(os.listdir(dir_path), desc=f'dictionary {dir_name}'):
            if '.D' in file_name or '.py' in file_name or '.json' in file_name:
                continue
            file_path = f'{dir_path}/{file_name}'
            if os.path.isdir(file_path):
                index = TextUtil.write_dictionary(dictionary_path=dictionary_path, dir_path=file_path, index=index,
                                                  add=add, select_id=select_id)
            else:
                TextUtil.write_dictionary_item(dictionary_path=dictionary_path, file_path=file_path,
                                               select_id=select_id,
                                               first_write=1 == index and add)
            index += 1
        return index

    @staticmethod
    def build_word_vector(dictionary_path, dir_path, vector_size, select_id, save_path):
        print(f'Build dictionary by {dir_path}')
        TextUtil.write_dictionary(dictionary_path=dictionary_path, dir_path=dir_path, select_id=select_id)
        print('Build word vector')
        model = Word2Vec(
            LineSentence(open(dictionary_path, 'r', encoding='utf-8')),
            sg=0,
            window=3,
            min_count=0,
            workers=8,
            vector_size=vector_size
        )
        model.save(save_path)
        return model

    @staticmethod
    def normal_words(words, word_size, embedding_model, china, padding='PADDING'):
        words = TextUtil.line_cut(line=words, china=china)
        words = [i for i in words.split(' ')]
        word_li = []
        index = 0
        while len(word_li) < word_size:
            if index < len(words):
                word = words[index]
                try:
                    word_wv = embedding_model.wv[word]
                except:
                    index += 1
                    continue
                index += 1
            else:
                word_wv = embedding_model.wv[padding]
            word_li.append(word_wv)
        return word_li

    @staticmethod
    def load(path):
        return Word2Vec.load(path)


if __name__ == '__main__':
    dictionary_path = 'temp.txt'
    dir_path = 'E:\IPM\weibo'
    vector_size = 100
    select_id = 6
    TextUtil.build_word_vector(dictionary_path=dictionary_path, dir_path=dir_path, vector_size=vector_size,
                               select_id=select_id)
