# -- coding:utf-8 --
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
k=128
ratings=pd.read_csv("/media/jxnu/Data/dataset/ml-100k/u2.base",sep="\t",names=["user_id","movie_id","rating","timestamp"])

test=pd.read_csv("/media/jxnu/Data/dataset/ml-100k/u2.test",sep="\t",names=["user_id","movie_id","rating","timestamp"])

n_users=np.max(ratings["user_id"])

n_movies=np.max(ratings["movie_id"])

print([n_users,n_movies,len(ratings)])

#plt.hist(ratings['rating'])
#plt.show()
#print np.mean(ratings['rating'])

model1=Sequential()

model1.add(Embedding(n_users+1,k,input_length=1))

model1.add(Reshape((k,)))

model2=Sequential()

model2.add(Embedding(n_movies+1,k,input_length=1))

model2.add(Reshape((k,)))

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

users=ratings['user_id'].values

movies=ratings["movie_id"].values

X_train=[users,movies]

Y_train=ratings['rating'].values

# model.fit(X_train,Y_train,batch_size=100,epochs=50)
model.load_weights('my_model_weight.h5')
# plot_model(model,to_file='model.png')
# model.save_weights('my_model_weight.h5')
# users_test=test['user_id'].values

# item_test=test['movie_id'].values
#
# rating_test=test['rating'].values

# model.save('my_model.h5')
# json_string=model.to_json()

sum=0

for i in range(test.shape[0]):
    sum+=(test['rating'][i]-model.predict([np.array([test['user_id'][i]]),np.array([test['movie_id'][i]])]))**2

mse=math.sqrt(sum/test.shape[0])

print "mse:",mse
timeend=datetime.datetime.now()

print "time:",(timeend-timestart).seconds