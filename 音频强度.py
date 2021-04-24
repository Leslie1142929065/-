import wave
import pyaudio
import numpy
import pylab

wf = wave.open(r"D:\王晨E\pythonProject\for_test\mysite\static\总\123\chunks\2021-04-21 15-40-30降噪.wav0003.wav", "rb")

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
nframes = wf.getnframes()
framerate = wf.getframerate()

str_data = wf.readframes(nframes)
wf.close()
wave_data = numpy.fromstring(str_data, dtype=numpy.short)
wave_data.shape = -1, 2
wave_data = wave_data.T
time = numpy.arange(0,nframes)*(1.0/framerate)
# # 绘制波形图
# pylab.plot(time, wave_data[0])
# pylab.subplot(212)
# pylab.plot(time, wave_data[1], c="g")
# pylab.xlabel("time (seconds)")
# pylab.show()
#绘制频谱图
N = 44100
start = 0  
df = framerate / (N - 1)
freq = [df * n for n in range(0, N)
wave_data2 = wave_data[0][start:start + N]
c = numpy.fft.fft(wave_data2) * 2 / N
d = int(len(c) / 2)
while freq[d] > 1500:
    d -= 10
pylab.plot(freq[:d - 1], abs(c[:d - 1]), 'r')
pylab.xlabel("amplitude frequency  Hz")
pylab.show()
