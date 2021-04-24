import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import keras
import pickle
import os
import librosa
from keras import layers
from keras import models
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization

from tensorflow.keras import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
path=r"D:\王晨E\pythonProject\for_test\mysite\static\files\test2.wav"
y,sr = librosa.load(path,sr=None)
def normalizeVoiceLen(y,normalizedLen):
    nframes=len(y)
    y = np.reshape(y,[nframes,1]).T
    #归一化音频长度为2s,32000数据点
    if(nframes<normalizedLen):
        res=normalizedLen-nframes
        res_data=np.zeros([1,res],dtype=np.float32)
        y = np.reshape(y,[nframes,1]).T
        y=np.c_[y,res_data]
    else:
        y=y[:,0:normalizedLen]
    return y[0]
def getNearestLen(framelength,sr):
    framesize = framelength*sr
    #找到与当前framesize最接近的2的正整数次方
    nfftdict = {}
    lists = [32,64,128,256,512,1024]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])#按与当前framesize差值升序排列
    framesize = int(sortlist[0][0])#取最接近当前framesize的那个2的正整数次方值为新的framesize
    return framesize
VOICE_LEN=32000
#获得N_FFT的长度
N_FFT=getNearestLen(0.25,sr)
#统一声音范围为前两秒
y=normalizeVoiceLen(y,VOICE_LEN)
print(y.shape)
#提取mfcc特征
mfcc_data=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13,n_fft=N_FFT,hop_length=int(N_FFT/4))
# 画出特征图，将MFCC可视化。转置矩阵，使得时域是水平的
plt.matshow(mfcc_data)
plt.title('MFCC')
counter=0
fileDirCASIA = r'D:\王晨E\pythonProject\for_test\mysite\static\CASIA database'
mfccs={}
mfccs['angry']=[]
mfccs['fear']=[]
mfccs['happy']=[]
mfccs['neutral']=[]
mfccs['sad']=[]
mfccs['surprise']=[]
# mfccs['disgust']=[]
listdir=os.listdir(fileDirCASIA)
for persondir in listdir:
    if(not r'.' in persondir):
        emotionDirName=os.path.join(fileDirCASIA,persondir)
        emotiondir=os.listdir(emotionDirName)
        for ed in emotiondir:
            if(not r'.' in ed):
                filesDirName=os.path.join(emotionDirName,ed)
                files=os.listdir(filesDirName)
                for fileName in files:
                    if(fileName[-3:]=='wav'):
                        counter+=1
                        fn=os.path.join(filesDirName,fileName)
                        print(str(counter)+fn)
                        y,sr = librosa.load(fn,sr=None)
                        y=normalizeVoiceLen(y,VOICE_LEN)#归一化长度
                        mfcc_data=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13,n_fft=N_FFT,hop_length=int(N_FFT/4))
                        feature=np.mean(mfcc_data,axis=0)
                        mfccs[ed].append(feature.tolist())
    with open('mfcc_feature_dict.pkl', 'wb') as f:
        pickle.dump(mfccs, f)
#读取特征
mfccs={}
with open('mfcc_feature_dict.pkl', 'rb') as f:
    mfccs=pickle.load(f)
 #设置标签
emotionDict={}
emotionDict['angry']=0
emotionDict['fear']=1
emotionDict['happy']=2
emotionDict['neutral']=3
emotionDict['sad']=4
emotionDict['surprise']=5
data=[]
labels=[]
data=data+mfccs['angry']
print(len(mfccs['angry']))
for i in range(len(mfccs['angry'])):
    labels.append(0)
data=data+mfccs['fear']
print(len(mfccs['fear']))
for i in range(len(mfccs['fear'])):
    labels.append(1)
print(len(mfccs['happy']))
data=data+mfccs['happy']
for i in range(len(mfccs['happy'])):
    labels.append(2)
print(len(mfccs['neutral']))
data=data+mfccs['neutral']
for i in range(len(mfccs['neutral'])):
    labels.append(3)
print(len(mfccs['sad']))
data=data+mfccs['sad']
for i in range(len(mfccs['sad'])):
    labels.append(4)
print(len(mfccs['surprise']))
data=data+mfccs['surprise']
for i in range(len(mfccs['surprise'])):
    labels.append(5)
print(len(data))
print(len(labels))
#设置数据维度
data=np.array(data)
data=data.reshape((data.shape[0],data.shape[1],1))
labels=np.array(labels)
labels=to_categorical(labels)
#数据标准化
DATA_MEAN=np.mean(data,axis=0)
DATA_STD=np.std(data,axis=0)
data-=DATA_MEAN
data/=DATA_STD
paraDict={}
paraDict['mean']=DATA_MEAN
paraDict['std']=DATA_STD
paraDict['emotion']=emotionDict
with open('mfcc_model_para_dict.pkl', 'wb') as f:
    pickle.dump(paraDict, f)
ratioTrain=0.8
numTrain=int(data.shape[0]*ratioTrain)
permutation = np.random.permutation(data.shape[0])
data = data[permutation,:]
labels = labels[permutation,:]
x_train=data[:numTrain]
x_val=data[numTrain:]
y_train=labels[:numTrain]
y_val=labels[numTrain:]
model = Sequential()
model.add(Conv1D(256, 5,padding='same',input_shape=(126,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same',))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(6))
model.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model.summary()
cnnhistory=model.fit(x_train, y_train, batch_size=16, epochs=300, validation_data=(x_val, y_val))
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_val,y_val,verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

y=normalizeVoiceLen(y,VOICE_LEN)#归一化长度
mfcc_data=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13,n_fft=N_FFT,hop_length=int(N_FFT/4))
feature=np.mean(mfcc_data,axis=0)
feature=feature.reshape((126,1))
feature-=DATA_MEAN
feature/=DATA_STD
feature=feature.reshape((1,126,1))
result=model.predict(feature)
index=np.argmax(result, axis=1)[0]
edr = dict([(i, t) for t, i in emotionDict.items()])
print(edr[index])

