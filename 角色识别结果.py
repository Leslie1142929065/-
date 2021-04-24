from keras.models import load_model
import pickle
import librosa
import numpy as np
import os
model = load_model(r"D:\王晨E\pythonProject\for_test\mysite\wavesegment\saved_models\role.1_Voice_Detection_Model.h5")#利用训练好的模型进行识别
paradict = {}
with open(r'D:\王晨E\pythonProject\for_test\mysite\wavesegment\mfcc_role.1_model_para_dict.pkl', 'rb') as f:
    paradict = pickle.load(f)
DATA_MEAN = paradict['mean']
DATA_STD = paradict['std']
emotionDict = paradict['emotion']
edr = dict([(i, t) for t, i in emotionDict.items()])
path =r"D:\王晨E\pythonProject\for_test\mysite\static\总\123\chunks\2021-04-21 15-40-30降噪.wav0013.wav"
y,sr = librosa.load(path,sr=None)
def normalizeVoiceLen(y,normalizedLen):
    nframes=len(y)
    y = np.reshape(y,[nframes,1]).T
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
    nfftdict = {}
    lists = [32,64]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])
    return framesize
VOICE_LEN=1920
N_FFT=getNearestLen(0.25,sr)
y, sr = librosa.load(path, sr=None)
y = normalizeVoiceLen(y, VOICE_LEN) 
mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
feature = np.mean(mfcc_data, axis=0)
feature = feature.reshape((121, 1))
feature -= DATA_MEAN
feature /= DATA_STD
feature = feature.reshape((1, 121, 1))
result = model.predict(feature)
index = np.argmax(result, axis=1)[0]
print(edr[index])
