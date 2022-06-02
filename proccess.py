

import sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
"""
   1、 问本抽取
"""
text_list = ["我是一名武汉纺织大学的学生", "我是一名研究生"]

dic = {
    "我":0,
    "是":1
}
# 2、text -> vector

vec_list = []
for text in text_list:
    vec = []
    for c in text:
        vec += [dic.get(c, 0)]
    vec_list.append(vec)
"""
    1 句子 13
    2 句子 7
    batch: 
"""
# 3、 确保 batch 里面的vec维度一致
dim = 20

# 4、 构建网络

class MyModel(layers.Layer):

    def __init__(self):
        super().__init__()

        self.embeddings = layers.Embedding(20, 128)
        self.bert = Bert(dim=128)
        self.lstm = layers.LSTM(128, return_sequences=True)  #
        self.linear = layers.Dense(16, activation="softmax")

    def call(self, x):
        x = self.embeddings(x)
        x = self.bert(x)
        x = self.lstm(x)
        x = self.linear(x)
        return x
# loss = tf.losses.categorical_crossentropy(y, y/)

optimizer.zero_step()

loss.backward()
optimizer.step()

for i in range(epochs):

    inputs = getNewData()
    y = model(inputs)
    y/
    with tf.GradientTape() as tape:
        loss = tf.losses.categorical_crossentropy(y, y /)
        loss.gradient()
        optimizer.apply_gradient()


