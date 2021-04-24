f = open(r'D:\王晨E\pythonProject\for_test\mysite\static\files\合并test.txt','r',encoding='UTF-8')
lines = f.readlines()
a = 0
for lines in lines:
   if "的" in lines:
    # a = a+1
    print(lines)
# print(a)