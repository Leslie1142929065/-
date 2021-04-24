import jieba
import matplotlib.pyplot as plt
# %matplotlib inline
import re
import collections
import time
import json
import wordcloud
import os

dirPath = r"D:\王晨E\pythonProject\for_test\mysite\static\总\123\总1.0" #所有txt位于的文件夹路径
files = os.listdir(dirPath)
res = ""
i = 0
for file in files:
    if file.endswith(".txt"):
        i += 1
        title = "第%s章 %s" % (i, file[0:len(file)-4])
        with open("D:/王晨E/pythonProject/for_test/mysite/static/总/123/总1.0/" + file, "r", encoding='utf-8') as file:
            content = file.read()
            file.close()
        append = "\n%s\n\n%s" % (title, content)
        res += append

with open(r"D:\王晨E\pythonProject\for_test\mysite\static\总\123\outfile.txt", "w", encoding='utf-8') as outFile:
    outFile.write(res)
    outFile.close()
print(len(res))

def fenci(file):
    f = open(file,encoding="utf-8")
    f_list = []
    stopword=[]
    #按回车读取文件
    for line in f.readlines():
        f_list.append(line)
    w = open(r'D:\王晨E\pythonProject\for_test\mysite\static\总\123\分词结果.txt','a+',encoding="utf-8")
    stop=open(r'D:\王晨E\pythonProject\for_test\mysite\static\总\123\cn_fencifabao.txt',encoding='utf-8').readlines()

    for each in stop: #对每一个停止词
        word=each.strip('\n') #去除停止词后面的换行符
        stopword.append(word) #把该停止词加入stopword列表
    print(stopword) #显示处理完的停止词列表

    for i in range(0,len(f_list)):
        split_words = [x for x in jieba.cut(f_list[i]) if x not in stopword and len(x)>1 and len(x)<6]
        split_words1 = list(filter(lambda x: not x.isdigit(), split_words))
        split_words2 = ' '.join(split_words)
        w.write(split_words2)
        w.write('\n')
    w.close()
#分词
t_start = time.time()
fenci(r'D:\王晨E\pythonProject\for_test\mysite\static\总\123\outfile.txt')
print('分词完成，用时%.3f 秒' % (time.time() - t_start))

w = open(r'D:\王晨E\pythonProject\for_test\mysite\static\总\123\分词结果.txt',encoding="utf-8").readlines()
print(w)

f = open(r'D:\王晨E\pythonProject\for_test\mysite\static\总\123\分词结果.txt',encoding="utf-8")
f_list = []
#按换行符\n 读取文件
aa=[]
for line in f.readlines():
    # data = line.rstrip("\n\t")
    line = line.strip('\n')#去掉行末尾的\n
    data = line.split('\n\t')
    for s in data:
        sub_str = s.split(' ')
        if sub_str:
            aa.append(sub_str)
# print(aa)
l = [n for a in aa for n in a ]#二维列表->一维列表
print(l[:20])
len(l)
word_counts = collections.Counter(l) # 对分词做词频统计
print(len(word_counts))
# print(word_counts)
word_counts_top = word_counts.most_common(10) # 获取前 100 最高频的词
len(word_counts_top)
top = dict(word_counts_top[0:10])
top

w= wordcloud.WordCloud(
    background_color="white", #背景颜色
    max_words=20, #显示最大词数
    font_path=r'D:\王晨E\pythonProject\for_test\mysite\static\总\123\MSYH.TTC', #使用字体
    min_font_size=5,
    max_font_size=100,
    width=800, #图幅宽度
    height=800
    )
cloud_text=",".join(l)
w.generate(cloud_text)
w.to_file("pic.png")
plt.figure(figsize=(10,10))
plt.imshow(w)
plt.axis("off")
plt.show()