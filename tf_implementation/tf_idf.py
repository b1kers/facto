import os
import csv
import nltk
import scipy
import string
# import stopwords
import numpy as np
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
nltk.download('punkt')

STOPWORDS = ['а', 'е', 'и', 'ж', 'м', 'о', 'на', 'не', 'ни', 'об', 'но', 'он', 'мне', 'мои', 'мож', 'она', 'они', 'оно', 'мной', 'много', 'многочисленное', 'многочисленная', 'многочисленные', 'многочисленный', 'мною', 'мой', 'мог', 'могут', 'можно', 'может', 'можхо', 'мор', 'моя', 'моё', 'мочь', 'над', 'нее', 'оба', 'нам', 'нем', 'нами', 'ними', 'мимо', 'немного', 'одной', 'одного', 'менее', 'однажды', 'однако', 'меня', 'нему', 'меньше', 'ней', 'наверху', 'него', 'ниже', 'мало', 'надо', 'один', 'одиннадцать', 'одиннадцатый', 'назад', 'наиболее', 'недавно', 'миллионов', 'недалеко', 'между', 'низко', 'меля', 'нельзя', 'нибудь', 'непрерывно', 'наконец', 'никогда', 'никуда', 'нас', 'наш', 'нет', 'нею', 'неё', 'них', 'мира', 'наша', 'наше', 'наши', 'ничего', 'начала', 'нередко', 'несколько', 'обычно', 'опять',
'около', 'мы', 'ну', 'нх', 'от', 'отовсюду', 'особенно', 'нужно', 'очень', 'отсюда', 'в', 'во', 'вон', 'вниз', 'внизу', 'вокруг', 'вот', 'восемнадцать', 'восемнадцатый', 'восемь', 'восьмой', 'вверх', 'вам', 'вами', 'важное', 'важная', 'важные', 'важный', 'вдали', 'везде', 'ведь', 'вас', 'ваш', 'ваша', 'ваше', 'ваши', 'впрочем', 'весь', 'вдруг', 'вы', 'все', 'второй', 'всем', 'всеми', 'времени', 'время', 'всему', 'всего', 'всегда', 'всех', 'всею', 'всю', 'вся', 'всё', 'всюду', 'г', 'год', 'говорил', 'говорит', 'года', 'году', 'где', 'да', 'ее', 'за', 'из', 'ли', 'же', 'им', 'до', 'по', 'ими', 'под', 'иногда', 'довольно', 'именно', 'долго', 'позже', 'более', 'должно', 'пожалуйста', 'значит', 'иметь', 'больше', 'пока', 'ему', 'имя', 'пор', 'пора', 'потом', 'потому', 'после', 'почему', 'почти', 'посреди', 'ей', 'два', 'две', 'двенадцать', 'двенадцатый', 'двадцать', 'двадцатый', 'двух', 'его', 'дел', 'или', 'без', 'день', 'занят', 'занята', 'занято', 'заняты', 'действительно', 'давно', 'девятнадцать', 'девятнадцатый', 'девять', 'девятый', 'даже', 'алло', 'жизнь', 'далеко', 'близко', 'здесь', 'дальше', 'для', 'лет', 'зато', 'даром', 'первый', 'перед', 'затем', 'зачем', 'лишь', 'десять', 'десятый', 'ею', 'её', 'их', 'бы', 'еще', 'при', 'был', 'про', 'процентов', 'против', 'просто', 'бывает', 'бывь', 'если', 'люди', 'была', 'были', 'было', 'будем', 'будет', 'будете', 'будешь', 'прекрасно', 'буду', 'будь', 'будто', 'будут', 'ещё', 'пятнадцать', 'пятнадцатый', 'друго', 'другое', 'другой', 'другие', 'другая', 'других', 'есть', 'пять', 'быть', 'лучше', 'пятый', 'к', 'ком', 'конечно', 'кому', 'кого', 'когда', 'которой', 'которого', 'которая', 'которые', 'который', 'которых', 'кем', 'каждое', 'каждая', 'каждые', 'каждый', 'кажется', 'как', 'какой', 'какая', 'кто', 'кроме', 'куда', 'кругом', 'с', 'т', 'у', 'я', 'та', 'те', 'уж', 'со', 'то', 'том', 'снова', 'тому', 'совсем', 'того', 'тогда', 'тоже', 'собой', 'тобой', 'собою', 'тобою', 'сначала', 'только', 'уметь', 'тот', 'тою', 'хорошо', 'хотеть', 'хочешь', 'хоть', 'хотя', 'свое', 'свои', 'твой', 'своей', 'своего', 'своих', 'свою', 'твоя', 'твоё', 'раз', 'уже', 'сам', 'там', 'тем', 'чем', 'сама', 'сами', 'теми', 'само', 'рано', 'самом', 'самому', 'самой', 'самого', 'семнадцать', 'семнадцатый', 'самим', 'самими', 'самих', 'саму', 'семь', 'чему', 'раньше', 'сейчас', 'чего', 'сегодня', 'себе', 'тебе', 'сеаой', 'человек', 'разве', 'теперь', 'себя', 'тебя', 'седьмой', 'спасибо', 'слишком', 'так', 'такое', 'такой', 'такие', 'также', 'такая', 'сих', 'тех', 'чаще', 'четвертый', 'через', 'часто', 'шестой', 'шестнадцать', 'шестнадцатый', 'шесть', 'четыре', 'четырнадцать', 'четырнадцатый', 'сколько', 'сказал', 'сказала', 'сказать', 'ту', 'ты', 'три', 'эта', 'эти',
'что', 'это', 'чтоб', 'этом', 'этому', 'этой', 'этого', 'чтобы', 'этот', 'стал', 'туда', 'этим', 'этими', 'рядом', 'тринадцать', 'тринадцатый', 'этих', 'третий', 'тут', 'эту', 'суть', 'чуть', 'тысяч']

def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

class TFIDF:

    def __init__(self):
        # Start a graph session or init new
        self.max_features = 1000
        self.text_data = None
        self.texts = None
        self.target = None
        self.tfidf = None
        self.sparse_tfidf_texts = None
        self.P = None
        self.read_file()
        self.normalize_text()
        self.tf_idf()
        
        # self.random_walk()

    # Define tokenizer
    def __tokenizer(text):
        # Private method
        words = nltk.word_tokenize(text)
        return words

    def read_file(self, csv_file='common_train.csv'):
        if os.path.isfile(csv_file):
            self.text_data = []
            with open(csv_file, 'r') as temp_output_file:
                reader = csv.reader(temp_output_file)
                for row in reader:
                    self.text_data.append(row)
        self.texts =  [x[1] for x in self.text_data]
        self.target = [x[0] for x in self.text_data]

    def lemmatize_texts(self):
        mystem = Mystem()
        lemmatized_texts = []
        for text in self.texts:
            lemmatized_texts.append(''.join(mystem.lemmatize(text)).rstrip())
        return lemmatized_texts

    def normalize_text(self):
        self.target = [1. if x == '1' else 0. for x in self.target]
        # Lower case
        self.texts = [x.lower() for x in self.texts]
        # Remove punctuation
        self.texts = [''.join(c for c in x if c not in string.punctuation) for x in self.texts]
        # Remove numbers
        self.texts = [''.join(c for c in x if c not in string.digits) for x in self.texts]
        # Trim extra whitespace
        self.texts = [' '.join(x.split()) for x in self.texts]
        # Lemmatize texts with mystem
        self.texts = self.lemmatize_texts()

    def tf_idf(self):
        # Create TF-IDF of texts
        # self.tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords.get_stopwords('ru'),
        self.tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=STOPWORDS,
                                max_features=self.max_features)
        texts = self.texts
        self.sparse_tfidf_texts = self.tfidf.fit_transform(texts)

    def random_walk(self):
        H_w = self.sparse_tfidf_texts
        H = np.zeros(H_w.shape, dtype='float64')
        cx = scipy.sparse.coo_matrix(H_w)
        for i, j, v in zip(cx.row, cx.col, cx.data):
            H[i][j] = 1. if v > 0 else 0.

        with tf.Session() as sess:
            Dv = tf.diag(np.array(sum(map(np.array, H))))
            Dvw = tf.diag(np.array([sum(x) for x in H_w.toarray()]))
            W = tf.diag(np.array([1. for x in H]))
            # Is equal to
            # self.P = np.linalg.inv(Dv) @ np.transpose(H) @ W @ np.linalg.inv(Dvw) @ H_w
            current = tf.matmul(tf.matrix_inverse(Dv), H, transpose_b=True).eval()
            current = tf.matmul(current, W).eval()
            current = tf.matmul(current, Dvw).eval()
            self.P = tf.matmul(current, H_w.toarray()).eval()

