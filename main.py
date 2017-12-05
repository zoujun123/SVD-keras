# -- coding:utf-8 --
#中文注释必备
import keras
from keras.models import Sequential
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Embedding,Dropout,Dense,Merge,Reshape
import time
import datetime
from keras.models import load_model
from keras.utils.vis_utils import plot_model





timestart=datetime.datetime.now()
#顺便记录下模型的运行时间
k=128

ratings=pd.read_csv("~/ml-100k/u1.base",sep="\t",names=["user_id","movie_id","rating","timestamp"])

test=pd.read_csv("~/ml-100k/u1.test",sep="\t",names=["user_id","movie_id","rating","timestamp"])
#分别导入训练集和测试集，设置好路径，

n_users=np.max(ratings["user_id"])

n_movies=np.max(ratings["movie_id"])
#获取做大索引

print([n_users,n_movies,len(ratings)])

#plt.hist(ratings['rating'])
#plt.show()
#print np.mean(ratings['rating'])
#这些可以查看评分的直方图的分布，评分的平均值
model1=Sequential()

model1.add(Embedding(n_users+1,k,input_length=1))

model1.add(Reshape((k,)))
#模型1，每一个用户表示一个成k维词向量，
model2=Sequential()

model2.add(Embedding(n_movies+1,k,input_length=1))

model2.add(Reshape((k,)))
#模型2，每一个电影表示成一个K维的词向量
model=Sequential()

model.add(Merge([model1,model2],mode='concat'))

model.add(Dropout(0.2))

model.add(Dense(k,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(int(k/4),activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(int(k/16),activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='adam')
#构造数层的神经网络模型，编译时loss选择均方误差，优化器’Adam‘

users=ratings['user_id'].values

movies=ratings["movie_id"].values

X_train=[users,movies]

Y_train=ratings['rating'].values

model.fit(X_train,Y_train,batch_size=100,epochs=50)
#训练模型，拟合训练集
sum=0

for i in range(test.shape[0]):
    sum+=(test['rating'][i]-model.predict([np.array([test['user_id'][i]]),np.array([test['movie_id'][i]])]))**2

mse=math.sqrt(sum/test.shape[0])
#测试，计算在测试集中的均方误差
print "mse:",mse
timeend=datetime.datetime.now()

print "time:",(timeend-timestart).seconds
#共使用的时间
