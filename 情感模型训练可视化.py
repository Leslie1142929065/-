from keras.models import load_model
import pickle
import librosa
import numpy as np

model = load_model(r'D:\王晨E\pythonProject\for_test\mysite\wavesegment\saved_models\Emotion_Voice_Detection_Model.h5')
paradict = {}
with open(r'D:\王晨E\pythonProject\for_test\mysite\wavesegment\mfcc_model_para_dict.pkl', 'rb') as f:
    paradict = pickle.load(f)
DATA_MEAN = paradict['mean']
DATA_STD = paradict['std']
emotionDict = paradict['emotion']
edr = dict([(i, t) for t, i in emotionDict.items()])
path =r"D:\王晨E\pythonProject\for_test\mysite\static\总\123\chunks\2021-04-21 15-40-30降噪.wav0056.wav"
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
y, sr = librosa.load(path, sr=None)
y = normalizeVoiceLen(y, VOICE_LEN)  # 归一化长度
mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
feature = np.mean(mfcc_data, axis=0)
feature = feature.reshape((126, 1))
feature -= DATA_MEAN
feature /= DATA_STD
feature = feature.reshape((1, 126, 1))
result = model.predict(feature)
index = np.argmax(result, axis=1)[0]
print(edr[index])