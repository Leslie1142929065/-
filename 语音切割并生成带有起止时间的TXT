import sys
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 初始化
# audiopath = r"D:\王晨E\pythonProject\for_test\mysite\static\files"
audiopath = "D:\王晨E\pythonProject\for_test\mysite\static\总\123"
# audiopath = r"D:\王晨E\pythonProject\for_test\mysite\static\files\test.mp3"
# f = os.listdir(audiopath)
# f.sort()
for a in os.listdir(audiopath):
    if a.endswith('wav'):
        mp3 = os.path.join(audiopath, a)
        break
# mp3 = os.listdir(audiopath)[0]
# mp3 = os.path.join(audiopath, mp3)
audiotype = 'wav'  # 如果wav、mp4其他格式参看pydub.AudioSegment的API
# 创建保存目录
filepath = os.path.split(audiopath)[0]
chunks_path = filepath + '/chunks/'
print(chunks_path)
# 读入音频
print('读入音频')
sound = AudioSegment.from_file(wav, format=audiotype)
# sound = sound[:3 * 60 * 1000]  # 如果文件较大，先取前3分钟测试，根据测试结果，调整参数
# 分割
print('开始分割')
chunks = split_on_silence(sound, min_silence_len=300,silence_thresh=-50)  # min_silence_len: 拆分语句时，静默满0.3秒则拆分。silence_thresh：小于-30dBFS以下的为静默。

if not os.path.exists(chunks_path): os.mkdir(chunks_path)
# 保存所有分段
print('开始保存')
for i in range(len(chunks)):
    new = chunks[i]
    save_name = chunks_path + str(a) + '%04d.%s' % (i, audiotype)
    new.export(save_name, format=audiotype)
    print('%04d' % i, len(new))
print('保存完毕')
print('*********************************************************')
print(len(chunks))
print('%04d' % (1), 0000, len(chunks[0]))
for i in range(1, len(chunks)):
    new = chunks[i - 1]
    chunks[i] += new
    # sys.stdout = Logger('a.txt')
    file = open('a.txt', "a")
    file.write('%04d' % i + ' ' + str(len(chunks[i - 1])) + ' ' + str(len(chunks[i])) + '\t\n')
    print('%04d' % (i + 1), len(chunks[i - 1]), len(chunks[i]))
print('****************************')
for i in range(1, len(chunks)):
    file_path = chunks_path
    file_name = file_path + '每段音频起始时间' +str(a) + '.txt'
    new = chunks[i - 1]
    chunks[i] += new
    file = open(file_name, 'a')
    file.write('%04d' % (i) + ' ' + str(len(chunks[i - 1])) + ' ' + str(len(chunks[i])) + '\t\n')
# chunks[i+1] += chunks[i]
# print(len(new), len(chunks[i+1]))
